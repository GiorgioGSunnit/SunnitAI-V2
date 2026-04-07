#!/bin/bash
# Updates LLM_BASE_URL in .env with the current active RunPod pod URL.
# Restarts the service only if the URL has changed.
#
# Usage:
#   ./scripts/update_runpod_url.sh <RUNPOD_API_KEY> <POD_ID>
#
# Or set environment variables and run without arguments:
#   export RUNPOD_API_KEY=your_key_here
#   export RUNPOD_POD_ID=your_pod_id_here
#   ./scripts/update_runpod_url.sh
#
# To run automatically every 5 minutes via cron:
#   crontab -e
#   Add this line:
#   */5 * * * * RUNPOD_API_KEY=your_key RUNPOD_POD_ID=your_pod_id /full/path/to/scripts/update_runpod_url.sh >> /var/log/runpod_url_update.log 2>&1

set -euo pipefail

RUNPOD_API_KEY=${1:-${RUNPOD_API_KEY:-}}
POD_ID=${2:-${RUNPOD_POD_ID:-}}
ENV_FILE=${ENV_FILE:-.env}

# Uncomment whichever restart method applies to your server:
# RESTART_CMD="systemctl restart your-service-name"
# RESTART_CMD="docker compose restart"

if [[ -z "$RUNPOD_API_KEY" || -z "$POD_ID" ]]; then
    echo "Error: RUNPOD_API_KEY and POD_ID are required"
    echo "Usage: $0 <api_key> <pod_id>"
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

echo "[$(date)] Fetching current URL for pod $POD_ID..."

RESPONSE=$(curl -s \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://api.runpod.io/graphql?query={pod(input:{id:\"$POD_ID\"}){runtime{ports{ip,privatePort,publicPort,type}}}}")

# Extract the public URL for port 8000
POD_URL=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    ports = data['data']['pod']['runtime']['ports']
    port = next((p for p in ports if p['privatePort'] == 8000), None)
    if port:
        print(f\"https://{port['ip']}-8000.proxy.runpod.net/v1\")
    else:
        print('')
except Exception as e:
    print('')
" 2>/dev/null)

if [[ -z "$POD_URL" ]]; then
    echo "[$(date)] Error: could not fetch pod URL. Is the pod running?"
    exit 1
fi

echo "[$(date)] Current pod URL: $POD_URL"

# Get the URL currently saved in .env
CURRENT_URL=$(grep "^LLM_BASE_URL=" "$ENV_FILE" | cut -d '=' -f2 || echo "")

if [[ "$CURRENT_URL" == "$POD_URL" ]]; then
    echo "[$(date)] URL unchanged, no restart needed."
    exit 0
fi

echo "[$(date)] URL has changed. Updating .env..."

# Update or insert LLM_BASE_URL in .env
if grep -q "^LLM_BASE_URL=" "$ENV_FILE"; then
    sed -i.bak "s|^LLM_BASE_URL=.*|LLM_BASE_URL=$POD_URL|" "$ENV_FILE"
else
    echo "LLM_BASE_URL=$POD_URL" >> "$ENV_FILE"
fi

echo "[$(date)] .env updated successfully."

# Restart the service if RESTART_CMD is set
if [[ -n "${RESTART_CMD:-}" ]]; then
    echo "[$(date)] Restarting service..."
    eval "$RESTART_CMD"
    echo "[$(date)] Service restarted."
else
    echo "[$(date)] RESTART_CMD not set — please restart the service manually."
fi

echo "[$(date)] Done."