#!/usr/bin/env bash
# =============================================================================
# SunnitAI ChatBot — Server Deployment Script
#
# Usage:
#   scp -r /path/to/ChatBot-1/app root@YOUR_SERVER:/opt/chatbot
#   scp deploy.sh env.production.template root@YOUR_SERVER:/opt/chatbot/
#   ssh root@YOUR_SERVER "bash /opt/chatbot/deploy.sh"
#   If .env is missing, deploy copies env.production.template → .env (edit secrets on server).
# =============================================================================

set -euo pipefail

APP_DIR="/opt/chatbot"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="chatbot"
PORT=8000
# Calculation platform: separate service on its own port (must differ from PORT,
# or the chatbot would POST /match to itself and every calculation silently
# falls back to RAG).
CALC_SERVICE_NAME="calc-platform"
CALC_PORT=8802

echo "=== SunnitAI ChatBot — Server Deployment ==="
echo ""

# -------------------------------------------------------
# 1. System dependencies
# -------------------------------------------------------
echo "[1/8] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip python3-dev build-essential > /dev/null 2>&1

# Fix locale warnings
if ! locale -a 2>/dev/null | grep -q "en_US.utf8"; then
    apt-get install -y -qq locales > /dev/null 2>&1
    sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen
    locale-gen > /dev/null 2>&1
fi
echo "  Python: $(python3 --version)"

# -------------------------------------------------------
# 2. Virtual environment
# -------------------------------------------------------
echo "[2/8] Setting up Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "  Existing venv found — recreating for clean install..."
    rm -rf "$VENV_DIR"
fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# -------------------------------------------------------
# 3. Install dependencies
# -------------------------------------------------------
echo "[3/8] Installing Python dependencies (this takes a few minutes)..."
cd "$APP_DIR"
pip install -e "." 2>&1 | tail -5
echo "  Dependencies installed"

# -------------------------------------------------------
# 4. Verify .env exists
# -------------------------------------------------------
echo "[4/8] Checking configuration..."
if [ ! -f "$APP_DIR/.env" ]; then
    if [ -f "$APP_DIR/env.production.template" ]; then
        echo "  No .env — creating from env.production.template (edit secrets before relying in prod)"
        cp "$APP_DIR/env.production.template" "$APP_DIR/.env"
    else
        echo ""
        echo "ERROR: .env file not found at $APP_DIR/.env"
        echo "Fix: cp $APP_DIR/env.production.template $APP_DIR/.env"
        echo "     (or copy your own .env.production to .env if you keep it on the server only)"
        exit 1
    fi
fi
echo "  .env found"

# -------------------------------------------------------
# 5. Quick smoke test — can Python import the app?
# -------------------------------------------------------
echo "[5/8] Smoke test — importing the app..."
cd "$APP_DIR"
if ! "$VENV_DIR/bin/python" -c "from src.chatbot.api import app; print('  Import OK')" 2>&1; then
    echo ""
    echo "ERROR: Failed to import the app. Check the logs above."
    echo "  Common fix: pip install missing packages manually"
    exit 1
fi

# -------------------------------------------------------
# 6. Run database migrations
# -------------------------------------------------------
echo "[6/8] Running database migrations..."
cd "$APP_DIR"
MIGRATION_DATABASE_URL="${MIGRATION_DATABASE_URL:-postgresql:///astrea}"
# Git checkouts may preserve an overly restrictive migrations directory mode.
# The postgres OS user needs read/execute access to load Alembic's Python files.
chmod -R a+rX "$APP_DIR/src/db/migrations"
chmod a+r "$APP_DIR/scripts/reconcile_alembic_version.py"
sudo -u postgres env MIGRATION_DATABASE_URL="$MIGRATION_DATABASE_URL" \
    "$VENV_DIR/bin/python" scripts/reconcile_alembic_version.py
sudo -u postgres env MIGRATION_DATABASE_URL="$MIGRATION_DATABASE_URL" \
    "$VENV_DIR/bin/alembic" upgrade head
echo "  Database migrations applied"

# -------------------------------------------------------
# 7. Create systemd service
# -------------------------------------------------------
echo "[7/8] Creating systemd services..."

# Calculation platform service (deterministic calculator engine). The chatbot
# reaches it over HTTP at CALC_PLATFORM_URL; it must be running and on its own
# port or calculator matching fails silently and requests fall back to RAG.
cat > /etc/systemd/system/${CALC_SERVICE_NAME}.service << EOF
[Unit]
Description=SunnitAI Calculation Platform
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$VENV_DIR/bin/python -m uvicorn calculation_platform.app.main:app --host 127.0.0.1 --port $CALC_PORT
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=SunnitAI ChatBot API
After=network.target ${CALC_SERVICE_NAME}.service
Wants=${CALC_SERVICE_NAME}.service

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
Environment="CALC_PLATFORM_URL=http://localhost:$CALC_PORT"
ExecStart=$VENV_DIR/bin/python -m uvicorn src.chatbot.api:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${CALC_SERVICE_NAME} > /dev/null 2>&1
systemctl enable ${SERVICE_NAME} > /dev/null 2>&1

# -------------------------------------------------------
# 8. Start the service and verify
# -------------------------------------------------------
echo "[8/8] Starting the calculation platform and chatbot services..."
systemctl restart ${CALC_SERVICE_NAME}
sleep 2
if systemctl is-active --quiet ${CALC_SERVICE_NAME} \
   && curl -sf http://localhost:${CALC_PORT}/health > /dev/null 2>&1; then
    echo "  Calculation platform healthy on port ${CALC_PORT}."
else
    echo "  WARNING: calculation platform did not come up on port ${CALC_PORT}."
    echo "  The calculator feature will be unavailable (requests fall back to RAG)."
    echo "  Check logs: journalctl -u ${CALC_SERVICE_NAME} --no-pager -n 30"
fi

systemctl restart ${SERVICE_NAME}

# Wait for startup
sleep 3

if systemctl is-active --quiet ${SERVICE_NAME}; then
    # Double-check with a health request
    if curl -sf http://localhost:${PORT}/api/health > /dev/null 2>&1; then
        echo ""
        echo "==========================================="
        echo "  DEPLOYMENT SUCCESSFUL"
        echo "==========================================="
        echo ""
        echo "  API running at:  http://$(hostname -I | awk '{print $1}'):${PORT}"
        echo "  Health check:    curl http://$(hostname -I | awk '{print $1}'):${PORT}/api/health"
        echo "  API docs:        http://$(hostname -I | awk '{print $1}'):${PORT}/docs"
        echo ""
        echo "  View logs:       journalctl -u ${SERVICE_NAME} -f"
        echo "  Restart:         systemctl restart ${SERVICE_NAME}"
        echo "  Stop:            systemctl stop ${SERVICE_NAME}"
    else
        echo ""
        echo "WARNING: Service is running but health check failed."
        echo "  It may still be starting up. Check in a few seconds:"
        echo "    curl http://localhost:${PORT}/api/health"
        echo "  Or check logs:"
        echo "    journalctl -u ${SERVICE_NAME} --no-pager -n 30"
    fi
else
    echo ""
    echo "ERROR: Service failed to start."
    echo ""
    echo "--- Last 30 lines of logs ---"
    journalctl -u ${SERVICE_NAME} --no-pager -n 30
    echo ""
    echo "--- Try running manually to see the full error ---"
    echo "  cd $APP_DIR && source venv/bin/activate"
    echo "  python -m uvicorn src.chatbot.api:app --host 0.0.0.0 --port $PORT"
    exit 1
fi
