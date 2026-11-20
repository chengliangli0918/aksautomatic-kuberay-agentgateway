#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" != "--confirm-cost" || $# != 1 ]]; then
  echo "Usage: ACR_NAME=<unique-name> CALLERS_FILE=<private-json-file> bash scripts/deploy.sh --confirm-cost" >&2
  echo "Optional: DNS_LABEL=<azure-dns-label> PUBLIC_HOSTNAME=<custom-dns-name> ACME_SERVER=<acme-directory-url>" >&2
  echo "Creates billable Azure resources, including an 8xA100 VM (~USD 40.962/hour GPU VM alone)," >&2
  echo "and a PUBLIC HTTPS endpoint that accepts any caller holding a configured API key." >&2
  exit 1
fi
: "${ACR_NAME:?Set a globally unique ACR_NAME}"
: "${CALLERS_FILE:?Set CALLERS_FILE to your private caller policy JSON file}"
[[ -r "$CALLERS_FILE" ]] || { echo "Cannot read CALLERS_FILE" >&2; exit 1; }
CALLERS_FILE="$(cd "$(dirname "$CALLERS_FILE")" && pwd)/$(basename "$CALLERS_FILE")"
for cmd in az kubectl kubelogin helm jq go curl sha256sum openssl; do
  command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 1; }
done
if [[ ! "$ACR_NAME" =~ ^[a-zA-Z0-9]{5,50}$ ]]; then
  echo "ACR_NAME must be 5-50 alphanumeric characters." >&2
  exit 1
fi
# Validate policies with the same parser used by the server, before any Azure changes.
CALLERS_FILE="$CALLERS_FILE" go test ./cmd/quota-proxy -run '^TestDeploymentCallers$' -count=1
bash scripts/preflight.sh
RESOURCE_GROUP="${RESOURCE_GROUP:-charlilirg}"
LOCATION="${LOCATION:-italynorth}"
CLUSTER_NAME="${CLUSTER_NAME:-aks-vllm-auto}"
subscription=$(az account show --query id -o tsv)
DNS_LABEL="${DNS_LABEL:-vllm-$(printf '%s' "$subscription/$RESOURCE_GROUP/$CLUSTER_NAME" | sha256sum | cut -c1-10)}"
azure_fqdn="$DNS_LABEL.$LOCATION.cloudapp.azure.com"
PUBLIC_HOSTNAME="${PUBLIC_HOSTNAME:-$azure_fqdn}"
ACME_SERVER="${ACME_SERVER:-https://acme-v02.api.letsencrypt.org/directory}"
if [[ ! "$DNS_LABEL" =~ ^[a-z][a-z0-9-]{1,61}[a-z0-9]$ ]]; then
  echo "DNS_LABEL must be 3-63 lowercase letters, digits, or hyphens and start with a letter." >&2
  exit 1
fi
if [[ ! "$PUBLIC_HOSTNAME" =~ ^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$ ]] || ((${#PUBLIC_HOSTNAME} > 253)); then
  echo "PUBLIC_HOSTNAME must be a lowercase DNS name." >&2
  exit 1
fi
if [[ ! "$ACME_SERVER" =~ ^https://[A-Za-z0-9.-]+(:[0-9]+)?(/[A-Za-z0-9._~-]+)*/?$ ]]; then
  echo "ACME_SERVER must be an https:// ACME directory URL." >&2
  exit 1
fi
image_tag=$(date -u +%Y%m%d%H%M%S)
image="$ACR_NAME.azurecr.io/quota-proxy:$image_tag"
clusters=$(az aks list --resource-group "$RESOURCE_GROUP" -o json)
if jq -e --arg name "$CLUSTER_NAME" 'any(.[]; .name == $name)' <<<"$clusters" >/dev/null; then
  jq -e --arg name "$CLUSTER_NAME" --arg location "$LOCATION" \
    'any(.[]; .name == $name and (.sku.name | ascii_downcase) == "automatic" and .location == $location)' \
    <<<"$clusters" >/dev/null || { echo "Existing cluster is not Automatic in the requested region." >&2; exit 1; }
else
  # Entra ID, Azure RBAC and disabled local accounts are Automatic defaults; they are set
  # explicitly because organizational policies can require them in the request itself.
  az aks create --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" \
    --location "$LOCATION" --sku automatic \
    --enable-aad --enable-azure-rbac --disable-local-accounts
fi
registries=$(az acr list --resource-group "$RESOURCE_GROUP" -o json)
if ! jq -e --arg name "$ACR_NAME" 'any(.[]; .name == $name)' <<<"$registries" >/dev/null; then
  az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" \
    --location "$LOCATION" --sku Basic --admin-enabled false
fi
az aks update --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" --attach-acr "$ACR_NAME"
az acr build --registry "$ACR_NAME" --image "quota-proxy:$image_tag" --platform linux/amd64 .
az acr build --registry "$ACR_NAME" --image "model-router:$image_tag" --platform linux/amd64 router

# Do not overwrite or use a shared kubeconfig/current-context.
work=$(mktemp -d)
pf_pid=""
cleanup() {
  if [[ -n "$pf_pid" ]]; then kill "$pf_pid" 2>/dev/null || true; fi
  rm -f "$work/kubeconfig" "$work/quota-proxy.yaml" "$work/model-router.yaml" "$work/edge.yaml" \
    "$work/bootstrap.crt" "$work/bootstrap.key" "$work/port-forward.log"
  rmdir "$work"
}
trap cleanup EXIT
export KUBECONFIG="$work/kubeconfig"
az aks get-credentials --subscription "$subscription" --resource-group "$RESOURCE_GROUP" \
  --name "$CLUSTER_NAME" --file "$KUBECONFIG"
# Automatic clusters use Entra ID; use Azure CLI tokens instead of interactive device-code sign-in.
kubelogin convert-kubeconfig -l azurecli --kubeconfig "$KUBECONFIG"
# Grant the deploying identity Kubernetes cluster-admin through Azure RBAC if needed. The
# object ID comes from the ARM token because Microsoft Graph can be blocked by conditional access.
cluster_id=$(az aks show --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" --query id -o tsv)
token_payload=$(az account get-access-token --query accessToken -o tsv | cut -d. -f2 | tr '_-' '/+')
while ((${#token_payload} % 4)); do token_payload+="="; done
deployer_oid=$(base64 -d <<<"$token_payload" | jq -r .oid)
deployer_type=$(az account show --query user.type -o tsv)
[[ "$deployer_type" == servicePrincipal ]] && principal_type=ServicePrincipal || principal_type=User
admin_role="Azure Kubernetes Service RBAC Cluster Admin"
if [[ "$(az role assignment list --scope "$cluster_id" --role "$admin_role" --include-inherited \
  --fill-principal-name false --query "length([?principalId=='$deployer_oid'])" -o tsv)" == 0 ]]; then
  az role assignment create --assignee-object-id "$deployer_oid" --assignee-principal-type "$principal_type" \
    --role "$admin_role" --scope "$cluster_id" -o none
fi
for _ in $(seq 1 30); do
  [[ "$(kubectl auth can-i '*' '*' --all-namespaces 2>/dev/null)" == yes ]] && break
  echo "Waiting for Kubernetes RBAC role assignment to propagate..."
  sleep 10
done
if [[ "$(kubectl auth can-i '*' '*' --all-namespaces 2>/dev/null)" != yes ]]; then
  echo "The deploying identity has no Kubernetes cluster-admin access after 5 minutes." >&2
  exit 1
fi
kubectl apply -f deploy/namespace.yaml

# AKS Automatic always enforces Deployment Safeguards and baseline Pod Security.
# Exclude only the namespaces that cannot comply: cert-manager (components
# without probes), inference-edge (ACME solver pods without probes) and
# gpu-system (the NVIDIA device plugin needs hostPath volumes). Existing
# exclusions are preserved.
safeguards=$(az aks safeguards show --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" -o json)
read -r -a excluded_ns <<<"$(jq -r '[(.properties.excludedNamespaces // .excludedNamespaces // [])[],
  "cert-manager", "gpu-system", "inference-edge"] | unique | join(" ")' <<<"$safeguards")"
az aks safeguards update --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" \
  --level Enforce --excluded-ns "${excluded_ns[@]}" -o none
# Exclusions propagate asynchronously. Wait until a probe-less pod is admitted, and in
# gpu-system also a hostPath volume (denied by AKS-managed baseline admission policies).
wait_for_safeguard_exclusion() {
  local deadline=$((SECONDS + 1800)) output pod
  pod=$(cat <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: safeguards-exclusion-check
spec:
  automountServiceAccountToken: false
  securityContext:
    runAsNonRoot: true
    runAsUser: 65532
    seccompProfile: {type: RuntimeDefault}
  containers:
    - name: check
      image: ghcr.io/agentgateway/agentgateway:v1.7.0@sha256:60216aafb03d37f94e952ff3d51d2f630a65b10ecaebdc0a864882f92610c45b
      resources:
        requests: {cpu: 100m, memory: 100Mi}
        limits: {cpu: 100m, memory: 100Mi}
      securityContext:
        allowPrivilegeEscalation: false
        capabilities: {drop: [ALL]}
EOF
  )
  if [[ "$1" == gpu-system ]]; then
    pod+=$'\n      volumeMounts: [{name: dp, mountPath: /dp, readOnly: true}]'
    pod+=$'\n  volumes: [{name: dp, hostPath: {path: /var/lib/kubelet/device-plugins}}]'
  fi
  until output=$(kubectl -n "$1" apply --dry-run=server -o name -f - <<<"$pod" 2>&1); do
    if ((SECONDS >= deadline)); then
      echo "Deployment Safeguards exclusion for namespace $1 is not active after 30 minutes:" >&2
      echo "$output" >&2
      return 1
    fi
    echo "Waiting for Deployment Safeguards exclusion in namespace $1..."
    sleep 30
  done
}
for ns in cert-manager gpu-system inference-edge; do
  wait_for_safeguard_exclusion "$ns"
done
helm upgrade --install kuberay-operator kuberay-operator \
  --repository-config /dev/null \
  --repo https://ray-project.github.io/kuberay-helm/ --version 1.7.2 \
  --namespace kuberay-system --create-namespace --wait --timeout 5m
if kubectl -n inference get rayservice vllm -o name --ignore-not-found | grep -q .; then
  echo "RayService already exists. Follow the README downtime procedure before redeploying." >&2
  exit 1
fi
helm upgrade --install nvidia-device-plugin nvidia-device-plugin \
  --repository-config /dev/null \
  --repo https://nvidia.github.io/k8s-device-plugin --version 0.20.2 \
  --namespace gpu-system --create-namespace -f deploy/nvidia-values.yaml --wait --timeout 5m
kubectl apply -f deploy/gpu-nodepool.yaml
kubectl -n inference create secret generic callers --from-file=callers.json="$CALLERS_FILE" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f deploy/network-policy.yaml
kubectl apply -f deploy/rayservice.yaml
sed "s|image: model-router:local|image: $ACR_NAME.azurecr.io/model-router:$image_tag|" \
  deploy/model-router.yaml > "$work/model-router.yaml"
kubectl apply -f "$work/model-router.yaml"
kubectl -n inference rollout status deployment/model-router --timeout=5m
sed "s|image: quota-proxy:local|image: $image|" deploy/quota-proxy.yaml > "$work/quota-proxy.yaml"
kubectl apply -f "$work/quota-proxy.yaml"
kubectl -n inference rollout status deployment/quota-proxy --timeout=5m

# Public HTTPS edge: agentgateway behind an Azure public IP, with a Let's Encrypt
# certificate issued and renewed by cert-manager over HTTP-01.
helm upgrade --install cert-manager cert-manager \
  --repository-config /dev/null \
  --repo https://charts.jetstack.io --version v1.21.3 \
  --namespace cert-manager -f deploy/cert-manager-values.yaml --wait --timeout 10m
kubectl -n inference-edge create configmap agentgateway-config --from-file=config.yaml=config/agentgateway.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
sed -e "s|__ACME_SERVER__|$ACME_SERVER|" -e "s|__PUBLIC_HOSTNAME__|$PUBLIC_HOSTNAME|" \
  -e "s|__DNS_LABEL__|$DNS_LABEL|" deploy/edge.yaml > "$work/edge.yaml"
if ! kubectl -n inference-edge get service agentgateway -o name --ignore-not-found | grep -q .; then
  available=$(az rest --method get --query available -o tsv --url \
    "https://management.azure.com/subscriptions/$subscription/providers/Microsoft.Network/locations/$LOCATION/CheckDnsNameAvailability?domainNameLabel=$DNS_LABEL&api-version=2024-05-01")
  if [[ "$available" != "true" ]]; then
    echo "DNS label '$DNS_LABEL' is already used in $LOCATION; set DNS_LABEL to another value." >&2
    exit 1
  fi
fi
if ! kubectl -n inference-edge get secret agentgateway-tls -o name --ignore-not-found | grep -q .; then
  # Self-signed bootstrap certificate so agentgateway can start and serve the
  # first HTTP-01 challenge. The issuer annotation deliberately differs from
  # the Issuer, so cert-manager keeps retrying until it replaces this Secret.
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 30 \
    -subj "/CN=$PUBLIC_HOSTNAME" -addext "subjectAltName=DNS:$PUBLIC_HOSTNAME" \
    -keyout "$work/bootstrap.key" -out "$work/bootstrap.crt" 2>/dev/null
  kubectl -n inference-edge create secret tls agentgateway-tls \
    --cert="$work/bootstrap.crt" --key="$work/bootstrap.key" --dry-run=client -o json |
    jq '.metadata.annotations = {"cert-manager.io/issuer-name": "bootstrap-self-signed",
      "cert-manager.io/issuer-kind": "Issuer", "cert-manager.io/issuer-group": "cert-manager.io"}' |
    kubectl create -f -
fi
kubectl apply -f "$work/edge.yaml"
kubectl -n inference-edge rollout status deployment/agentgateway --timeout=5m
public_ip=""
for _ in $(seq 1 60); do
  public_ip=$(kubectl -n inference-edge get service agentgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  [[ -n "$public_ip" ]] && break
  sleep 10
done
[[ -n "$public_ip" ]] || { echo "Timed out waiting for the gateway public IP." >&2; exit 1; }
echo "Gateway public IP: $public_ip (Azure DNS name: $azure_fqdn)"
if [[ "$PUBLIC_HOSTNAME" != "$azure_fqdn" ]]; then
  echo "Create this DNS record now: $PUBLIC_HOSTNAME CNAME $azure_fqdn"
fi
cert_deadline=$((SECONDS + 1800))
until kubectl -n inference-edge get certificate agentgateway-tls -o json | jq -e '(.status.revision // 0) >= 1 and
  any(.status.conditions[]?; .type == "Ready" and .status == "True")' >/dev/null; do
  if ((SECONDS >= cert_deadline)); then
    kubectl -n inference-edge describe certificate,certificaterequest,order,challenge >&2 || true
    echo "No trusted certificate after 30 minutes; HTTP-01 needs public port 80 and DNS for $PUBLIC_HOSTNAME." >&2
    echo "cert-manager keeps retrying with backoff; see README to force an immediate retry." >&2
    exit 1
  fi
  sleep 15
done
# Verify the trusted certificate and routing through a port-forward, which also works from
# networks whose egress filtering blocks the public IP; then try the public URL directly.
kubectl -n inference-edge port-forward service/agentgateway :443 >"$work/port-forward.log" 2>&1 &
pf_pid=$!
local_port=""
for _ in $(seq 1 30); do
  local_port=$(sed -n 's/^Forwarding from 127\.0\.0\.1:\([0-9]*\) .*/\1/p' "$work/port-forward.log" | head -1)
  [[ -n "$local_port" ]] && break
  sleep 1
done
[[ -n "$local_port" ]] || { echo "kubectl port-forward to the gateway failed." >&2; exit 1; }
# The kubelet can take a minute to project the issued Secret; curl verifies the chain.
status=""
for _ in $(seq 1 30); do
  status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
    --connect-to "$PUBLIC_HOSTNAME:443:127.0.0.1:$local_port" "https://$PUBLIC_HOSTNAME/v1/models" || true)
  [[ "$status" == 401 ]] && break
  sleep 10
done
kill "$pf_pid" 2>/dev/null || true
pf_pid=""
if [[ "$status" != 401 ]]; then
  echo "Gateway check failed: expected HTTP 401 without an API key over trusted TLS, got '$status'." >&2
  exit 1
fi
direct=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$PUBLIC_HOSTNAME/v1/models" || true)
if [[ "$direct" != 401 ]]; then
  echo "Warning: this machine cannot reach https://$PUBLIC_HOSTNAME directly (got '$direct')." >&2
  echo "Let's Encrypt reached it from the internet; check for a VPN that tunnels this IP range, or egress filtering." >&2
fi

kubectl -n inference wait --for=jsonpath='{.status.serviceStatus}'=Running rayservice/vllm --timeout=45m
kubectl -n inference rollout restart deployment/quota-proxy
kubectl -n inference rollout status deployment/quota-proxy --timeout=5m
echo "Ready: https://$PUBLIC_HOSTNAME/v1 is PUBLIC; requests need a caller API key (see README)."
