#!/usr/bin/env bash
set -euo pipefail

for cmd in az jq curl; do
  command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 1; }
done
RESOURCE_GROUP="${RESOURCE_GROUP:-charlilirg}"
LOCATION="${LOCATION:-italynorth}"
GPU_SKU=Standard_ND96amsr_A100_v4

az account show --query '{subscription:name,id:id}' -o json
az group show --name "$RESOURCE_GROUP" --query '{name:name,location:location}' -o json
echo "Checking $GPU_SKU in $LOCATION (SKU listing can take several minutes)..."
skus=$(az vm list-skus --location "$LOCATION" --resource-type virtualMachines \
  --size "$GPU_SKU" --all -o json)
sku=$(jq -ce --arg name "$GPU_SKU" '.[] | select(.name == $name)' <<<"$skus")
jq -e '(.restrictions | length) == 0' <<<"$sku" >/dev/null || {
  jq '.restrictions' <<<"$sku" >&2
  echo "Requested GPU SKU has subscription restrictions; refusing deployment." >&2
  exit 1
}
family=$(jq -r '.family | ascii_downcase' <<<"$sku")
usage=$(az vm list-usage --location "$LOCATION" -o json)
jq -e --arg family "$family" '
  any(.[]; (.name.value | ascii_downcase) == $family and
    ((.limit | tonumber) - (.currentValue | tonumber)) >= 96) and
  any(.[]; .name.value == "cores" and
    ((.limit | tonumber) - (.currentValue | tonumber)) >= 112)
' <<<"$usage" >/dev/null || {
  echo "Need 96 free GPU-family vCPUs and at least 112 free regional vCPUs." >&2
  jq --arg family "$family" '.[] | select((.name.value | ascii_downcase) == $family or .name.value == "cores")' <<<"$usage" >&2
  exit 1
}
curl --fail --silent --show-error --get 'https://prices.azure.com/api/retail/prices' \
  --data-urlencode "\$filter=armRegionName eq '$LOCATION' and armSkuName eq '$GPU_SKU' and priceType eq 'Consumption'" |
  jq --arg sku "$GPU_SKU" '[.Items[] | select(.skuName == $sku and (.productName | endswith("Linux"))) |
    {currency: .currencyCode, hourly: .retailPrice, monthly_730h: (.retailPrice * 730), unit: .unitOfMeasure}]'
echo "Quota checks passed. SKU listing is NOT a capacity reservation or allocation guarantee."
