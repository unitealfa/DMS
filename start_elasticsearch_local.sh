#!/bin/bash

echo "[INFO] Demarrage Elasticsearch..."

sudo systemctl start elasticsearch

echo
echo "[INFO] Elasticsearch lance"
echo "Verification : http://localhost:9200"

sleep 3
xdg-open http://localhost:9200