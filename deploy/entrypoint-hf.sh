#!/bin/sh
# HF Spaces entrypoint: single container, single port ($PORT, default 7860).
set -e

: "${PORT:=7860}"
export PORT

cd /app

# Backend API
python -m uvicorn backend_app.app:app --host 127.0.0.1 --port 8200 --workers 1 &

# Next.js standalone server
cd /app/frontend
HOSTNAME=127.0.0.1 PORT=3000 node server.js &
cd /app

# Caddy fronts both on the public port (keeps the container alive)
exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
