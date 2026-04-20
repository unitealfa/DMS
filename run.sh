#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

API_BASE_IMAGE="python:3.11-slim"
ES_IMAGE="docker.elastic.co/elasticsearch/elasticsearch:8.13.4"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  echo "Docker n'est pas accessible sur cette machine." >&2
  exit 1
fi

compose() {
  "${DOCKER[@]}" compose "$@"
}

docker_cmd() {
  "${DOCKER[@]}" "$@"
}

image_present() {
  docker_cmd image inspect "$1" >/dev/null 2>&1
}

print_docker_network_fix() {
  cat >&2 <<EOF
Docker n'arrive pas a joindre le registre distant depuis cette machine.

Cause probable:
- Docker Hub inaccessible
- IPv6/DNS cassé côté daemon Docker
- réseau machine/box/proxy qui bloque la sortie du daemon

Lance ces commandes sur le PC concerné puis réessaie:

sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1
sudo systemctl restart docker
docker pull ${API_BASE_IMAGE}
docker pull ${ES_IMAGE}
cd ~/Bureau/DMS
./run.sh
EOF
}

pull_image_if_needed() {
  local image="$1"
  local label="$2"
  local log_file
  if image_present "$image"; then
    return 0
  fi

  log_file="$(mktemp)"
  echo "==> Préchargement image ${label}: ${image}"
  if docker_cmd pull "$image" >"$log_file" 2>&1; then
    cat "$log_file"
    rm -f "$log_file"
    return 0
  fi

  cat "$log_file" >&2
  if grep -qiE 'failed to fetch anonymous token|auth\.docker\.io|network is unreachable|TLS handshake timeout|i/o timeout|Temporary failure in name resolution' "$log_file"; then
    print_docker_network_fix
  fi
  rm -f "$log_file"
  return 1
}

pull_image_if_needed "$API_BASE_IMAGE" "base Python"
pull_image_if_needed "$ES_IMAGE" "Elasticsearch"

echo "==> Build + start des conteneurs..."
compose up -d --build

echo "==> Attente du démarrage de l'API..."
for _ in $(seq 1 30); do
  if curl -fsS http://localhost:8765/api/status >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "==> Vérification de l'API..."
curl -fsS http://localhost:8765/api/status || true

echo
echo "Stack lancée."
echo "API: http://localhost:8765"
echo "Status: http://localhost:8765/api/status"
echo
echo "Mode dev Docker actif: ./core est monté en volume dans /app."
echo "- Les modifications dans core/ sont visibles directement dans le conteneur."
echo "- Pour du code uniquement, pas besoin de rebuild; redémarre juste dms-api."
echo
echo "Relance rapide API après modification de code:"
echo "${DOCKER[*]} restart dms-api"
echo
echo "Commandes directes disponibles depuis la racine du dépôt:"
echo "python main.py documents/image2tab.webp --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens"
echo "python local_api.py --host 0.0.0.0 --port 8765"
echo
echo "Logs API:"
echo "${DOCKER[*]} logs -f dms-api"
echo
echo "Ouvrir un shell dans le conteneur:"
echo "${DOCKER[*]} exec -it dms-api bash"
echo
echo "Stop:"
echo "${DOCKER[*]} compose down"
