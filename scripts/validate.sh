#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AGENTGATEWAY_BIN:?Set AGENTGATEWAY_BIN to the v1.7.0 binary}"
command -v openssl >/dev/null || { echo "Missing command: openssl" >&2; exit 1; }
for file in scripts/*.sh; do bash -n "$file"; done
unformatted=$(gofmt -l cmd)
[[ -z "$unformatted" ]] || { echo "Run gofmt on: $unformatted" >&2; exit 1; }
go vet ./cmd/quota-proxy
# Includes TestPublicGatewayConfig, which runs config/agentgateway.yaml in the real binary.
go test -race -count=1 ./cmd/quota-proxy
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 1 \
  -subj /CN=validate.invalid -keyout "$tmp/tls.key" -out "$tmp/tls.crt" 2>/dev/null
sed "s|/tls/|$tmp/|g" config/agentgateway.yaml > "$tmp/config.yaml"
"$AGENTGATEWAY_BIN" --validate-only -f "$tmp/config.yaml"
echo "Local checks passed. Kubernetes server-side validation still requires an AKS cluster and installed CRDs."
