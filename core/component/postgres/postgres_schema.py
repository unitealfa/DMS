"""
Configuration de la base cible PostgreSQL et de son schema.

Tu controles ici :
- le nom exact de la base a creer si elle n'existe pas
- les extensions optionnelles
- les tables a creer automatiquement si elles manquent
- les index utiles pour les futures insertions / recherches
- les ALTER TABLE de rattrapage si un schema plus ancien existe deja
"""

POSTGRES_DATABASE_NAME = "dms_core"

POSTGRES_EXTENSIONS = []

POSTGRES_TABLES = [
    {
        "name": "dms_pipeline_launch_logs",
        "sql": """
        CREATE TABLE IF NOT EXISTS dms_pipeline_launch_logs (
            log_id TEXT PRIMARY KEY,
            run_id TEXT,
            pipeline_profile TEXT,
            filename TEXT,
            status_text TEXT,
            launched_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
]

POSTGRES_INDEXES = [
    {
        "name": "idx_dms_pipeline_launch_logs_run_id",
        "sql": "CREATE INDEX IF NOT EXISTS idx_dms_pipeline_launch_logs_run_id ON dms_pipeline_launch_logs (run_id);",
    },
    {
        "name": "idx_dms_pipeline_launch_logs_filename",
        "sql": "CREATE INDEX IF NOT EXISTS idx_dms_pipeline_launch_logs_filename ON dms_pipeline_launch_logs (filename);",
    },
    {
        "name": "idx_dms_pipeline_launch_logs_logged_at",
        "sql": "CREATE INDEX IF NOT EXISTS idx_dms_pipeline_launch_logs_logged_at ON dms_pipeline_launch_logs (launched_at DESC);",
    },
]

POSTGRES_POST_SQL = []
