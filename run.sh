#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "==> Build + start des conteneurs..."
sudo docker compose up -d --build

echo "==> Attente du démarrage de l'API..."
sleep 8

echo "==> Vérification de l'API..."
curl http://localhost:8765/api/status || true

echo
echo "Stack lancée."
echo "API: http://localhost:8765"
echo "Status: http://localhost:8765/api/status"
echo
echo "Logs API:"
echo "sudo docker logs -f dms-api"
echo
echo "Stop:"
echo "sudo docker compose down"
