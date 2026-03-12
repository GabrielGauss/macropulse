#!/usr/bin/env bash
# ============================================================
# MacroPulse VPS Setup Script
# Run once on a fresh Ubuntu 22.04 / 24.04 server.
#
#   curl -fsSL https://raw.githubusercontent.com/GabrielGauss/macropulse/master/scripts/setup_vps.sh | bash
#
# Prerequisites:
#   - DNS A record: api.macropulse.live → this server's IP
#   - .env file ready to upload (see .env.example)
# ============================================================
set -euo pipefail

DOMAIN="api.macropulse.live"
REPO="https://github.com/GabrielGauss/macropulse.git"
APP_DIR="/opt/macropulse"
EMAIL="support@macropulse.live"   # Let's Encrypt notifications

echo "==> [1/7] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq curl git ufw

echo "==> [2/7] Installing Docker..."
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
fi

echo "==> [3/7] Configuring firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo "==> [4/7] Cloning repository..."
if [ -d "$APP_DIR" ]; then
  echo "    Directory exists — pulling latest..."
  git -C "$APP_DIR" pull origin master
else
  git clone "$REPO" "$APP_DIR"
fi

echo "==> [5/7] Checking .env file..."
if [ ! -f "$APP_DIR/.env" ]; then
  echo ""
  echo "    *** ACTION REQUIRED ***"
  echo "    Upload your .env file to $APP_DIR/.env before continuing."
  echo "    Example: scp .env root@<server-ip>:/opt/macropulse/.env"
  echo ""
  echo "    Then re-run: cd $APP_DIR && bash scripts/setup_vps.sh"
  exit 1
fi

echo "==> [6/7] Obtaining SSL certificate (Let's Encrypt)..."
cd "$APP_DIR"

# Start nginx on HTTP only first (needs to be up for ACME challenge)
# Use a temporary minimal config
cat > /tmp/nginx-init.conf << 'NGINXEOF'
server {
    listen 80;
    server_name api.macropulse.live;
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 200 'ok'; }
}
NGINXEOF

docker run -d --rm --name nginx-init \
  -p 80:80 \
  -v /tmp/nginx-init.conf:/etc/nginx/conf.d/default.conf:ro \
  -v certbot_www:/var/www/certbot \
  nginx:alpine 2>/dev/null || true

sleep 2

# Get the cert
docker run --rm \
  -v certbot_conf:/etc/letsencrypt \
  -v certbot_www:/var/www/certbot \
  certbot/certbot certonly \
  --webroot -w /var/www/certbot \
  --non-interactive --agree-tos \
  -m "$EMAIL" \
  -d "$DOMAIN"

# Stop temporary nginx
docker stop nginx-init 2>/dev/null || true

echo "==> [7/7] Starting MacroPulse stack..."
docker compose up -d

echo ""
echo "============================================"
echo "  MacroPulse is live!"
echo "  API:     https://$DOMAIN/health"
echo "  Docs:    https://$DOMAIN/docs"
echo "  Dashboard: https://$DOMAIN/dashboard"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Run the backfill: docker compose exec api python scripts/backfill_history.py --start 2023-01-01 --version v2"
echo "  2. Check logs:       docker compose logs -f api"
