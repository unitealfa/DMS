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
sudo docker restart dms-api
```

Rebuild requis seulement si tu modifies:
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`

Dans ce cas:
```bash
sudo docker compose up -d --build
```

## API
- base URL: `http://127.0.0.1:8765`
- status: `http://127.0.0.1:8765/api/status`

Pour le reste:
- lire `core/README.md`
