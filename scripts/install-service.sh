#!/bin/bash
set -e

if [ "$EUID" -ne 0 ]; then
  echo "Run as root: sudo bash scripts/install-service.sh"
  exit 1
fi

BINARY="/usr/local/bin/website-serve"
SERVICE="website-serve"

echo "Building binary..."
bash "$(dirname "$0")/build-binary.sh"

echo "Installing binary to $BINARY..."
cp website-serve "$BINARY"
chmod +x "$BINARY"

echo "Verifying binary..."
ls -lh "$BINARY"
file "$BINARY"
"$BINARY" -h 2>&1 | head -2 && echo "  (binary runs OK)" || echo "  (WARNING: binary failed to execute — check architecture)"

echo "Creating systemd service..."
cat > /etc/systemd/system/$SERVICE.service << UNIT
[Unit]
Description=website-serve — SPA static server (backup binary)
Documentation=https://kgup.me
After=network.target

[Service]
Type=simple
User=root
Group=root
ExecStart=$BINARY
Restart=always
RestartSec=5
Environment=PORT=8080

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable $SERVICE
systemctl restart $SERVICE

echo ""
echo "Done. Service status:"
systemctl status $SERVICE --no-pager

echo ""
echo "Manage it:"
echo "  sudo systemctl status website-serve"
echo "  sudo systemctl restart website-serve"
echo "  sudo journalctl -u website-serve -f"
echo ""
echo "To switch nginx to proxy to it, set in your nginx config:"
echo "  proxy_pass http://127.0.0.1:8080;"
