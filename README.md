# DMS

Le code applicatif principal est dans `core/`.

Documentation complete:
- `core/README.md`

## Demarrage rapide Docker
Depuis la racine du depot:
```bash
cd ~/Bureau/DMS
./run.sh
```

Ce que fait `run.sh`:
- build la stack Docker
- demarre `dms-api`
- demarre `dms-elasticsearch`
- expose l'API sur `http://127.0.0.1:8765`
- force le mode offline fallback pour eviter les downloads de modeles HF dans Docker

## Commandes utiles Docker
Ajoute `sudo` devant `docker` si ta machine le demande.

Premier lancement:
```bash
cd ~/Bureau/DMS
./run.sh
```

Rebuild complet apres changement de `Dockerfile`, `docker-compose.yml` ou `requirements.txt`:
```bash
cd ~/Bureau/DMS
docker compose up -d --build
```

Redemarrage rapide de l'API apres changement de code dans `core/`:
```bash
cd ~/Bureau/DMS
docker compose restart dms-api
```

Verifier que l'API tourne et voir la pipeline active:
```bash
curl -s http://127.0.0.1:8765/api/status
```

Suivre les logs de l'API:
```bash
docker logs -f dms-api
```

Arreter toute la stack:
```bash
cd ~/Bureau/DMS
docker compose down
```

## Commandes directes depuis la racine du depot
Une fois la stack Docker lancee, tu peux utiliser directement:

```bash
python main.py documents/image2tab.webp --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens
```

```bash
python local_api.py --host 0.0.0.0 --port 8765
```

Comportement:
- si l'environnement Python local est pret, ces commandes tournent en local
- sinon, elles deleguent automatiquement au conteneur Docker `dms-api`

## Modifications de code en mode Docker
Le dossier `./core` est monte en volume dans le conteneur sur `/app`.

Donc:
- si tu modifies seulement du code dans `core/`, pas besoin de rebuild
- redemarre juste le service API:
```bash
docker compose restart dms-api
```

Rebuild requis seulement si tu modifies:
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`

Dans ce cas:
```bash
docker compose up -d --build
```

## API
- base URL: `http://127.0.0.1:8765`
- status: `http://127.0.0.1:8765/api/status`

Pour le reste:
- lire `core/README.md`
