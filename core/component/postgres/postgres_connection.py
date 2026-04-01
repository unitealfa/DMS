"""
Configuration centralisee PostgreSQL pour le pipeline DMS.

Tu controles ici :
- l'activation globale PostgreSQL
- la connexion host/port/user/password
- le demarrage automatique local si PostgreSQL n'est pas joignable
- l'activation de la synchro finale vers la base
- le fait d'ecrire un audit `postgres_sync` dans `fusion_output.json`
- pour l'instant, la synchro ecrit seulement un log minimal
  `temps + nom du fichier + ok` dans PostgreSQL

Le nom de la base et le schema des tables sont geres dans :
`component/postgres/postgres_schema.py`
"""

POSTGRES_ENABLED = True

# Si True, un echec de bootstrap PostgreSQL casse le lancement.
# Si False, le pipeline continue mais journalise l'erreur.
POSTGRES_STRICT_BOOTSTRAP = False

# Connexion PostgreSQL locale par defaut.
POSTGRES_HOST = "127.0.0.1"
POSTGRES_PORT = 5432
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "8425"

# Base d'administration utilisee pour verifier / creer la base cible.
POSTGRES_ADMIN_DATABASE = "postgres"

# Outils PostgreSQL utilises par le bootstrap.
POSTGRES_PSQL_BIN = "psql"
POSTGRES_PG_ISREADY_BIN = "pg_isready"
POSTGRES_CONNECT_TIMEOUT = 5

# Demarrage auto du service si PostgreSQL n'est pas joignable.
POSTGRES_AUTO_START = True
POSTGRES_AUTO_START_WAIT_SECONDS = 45
POSTGRES_AUTO_START_LAUNCH_TIMEOUT = 20

# Optionnel : si tu utilises `sudo -S ...` dans une commande de start.
POSTGRES_START_PASSWORD = ""

# Ordre de tentative des commandes de demarrage.
# Tu peux les adapter librement a ta machine.
POSTGRES_START_COMMANDS = [
    "systemctl start postgresql",
    "service postgresql start",
    "pg_ctlcluster 16 main start",
    "pg_ctlcluster 15 main start",
    "pg_ctlcluster 14 main start",
    "docker start postgres",
    "docker compose up -d postgres",
    "docker-compose up -d postgres",
]

# Synchronisation finale vers PostgreSQL.
# Pour l'instant le composant n'envoie qu'un log minimal par document
# dans `dms_pipeline_launch_logs` (date + fichier + status `ok`).
POSTGRES_SYNC_ENABLED = True
POSTGRES_SYNC_STRICT = False
POSTGRES_SYNC_WRITE_FUSION_AUDIT = True
POSTGRES_SYNC_UPSERT_RUNS = True
POSTGRES_SYNC_UPSERT_DOCUMENTS = True
POSTGRES_SYNC_UPSERT_LINKS = True

# Fichier schema / base cible.
POSTGRES_SCHEMA_CONFIG = "component/postgres/postgres_schema.py"
