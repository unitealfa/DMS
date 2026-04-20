#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

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
