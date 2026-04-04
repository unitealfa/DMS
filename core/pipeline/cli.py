from __future__ import annotations

import argparse
import logging
import builtins
import os
from pathlib import Path

from .orchestrator import Pipeline0MLOrchestrator, Pipeline50MLOrchestrator, Pipeline100MLOrchestrator
from .postgres import ensure_postgres_bootstrap
from .settings import configure_logging, load_dotenv, normalize_input

# Pipeline par defaut configurable directement dans le code.
# Valeurs supportees: "pipeline0ml" | "pipeline50ml" | "pipeline100ml"
PIPELINE_DEFAULT_CODE = "pipeline0ml"

_STEP_ALIASES = {
    "atripusion-gramatical-en-utilisant-les3ficherla": "atripusion-gramatical",
}

_STEP_CHOICES = [
    "pretraitement-de-docs",
    "si-image-pretraiter-sinonpass-le-doc",
    "output-txt",
    "clasification",
    "tokenisation-layout",
    "atripusion-gramatical",
    "atripusion-gramatical-en-utilisant-les3ficherla",
    "table-extraction",
    "verification-totaux",
    "detection-signature-chachet-codebarr",
    "liaison-inter-docs",
    "elasticsearch",
    "extraction-regles",
    "fusion-resultats",
    "postgres-sync",
]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_optional(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _normalize_pipeline_name(raw: str | None, default: str = "pipeline0ml") -> str:
    aliases = {
        "default": "pipeline0ml",
        "pipeline0ml": "pipeline0ml",
        "0ml": "pipeline0ml",
        "pipeline50ml": "pipeline50ml",
        "50ml": "pipeline50ml",
        "pipeline100ml": "pipeline100ml",
        "100ml": "pipeline100ml",
    }
    if raw is None:
        return default
    value = str(raw).strip().lower()
    return aliases.get(value, default)


def _env_pipeline(default: str | None = None) -> str:
    base_default = _normalize_pipeline_name(default or PIPELINE_DEFAULT_CODE, "pipeline0ml")
    raw = os.environ.get("PIPELINE_DEFAULT")
    if raw is None:
        raw = os.environ.get("PIPELINE_PROFILE")
    return _normalize_pipeline_name(raw, base_default)


def _normalize_step_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    return _STEP_ALIASES.get(value, value)


def parse_cli() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env", override=False)

    parser = argparse.ArgumentParser(description="Orchestrateur du pipeline documentaire.")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Chemins des fichiers a traiter (separes par des virgules ou des espaces).",
    )
    parser.add_argument(
        "--pipeline",
        choices=["default", "pipeline0ml", "pipeline50ml", "pipeline100ml"],
        default=_env_pipeline(),
        help=(
            "Pipeline a executer: default/pipeline0ml (pipeline actuelle) ou pipeline50ml "
            "(tokenisation/extraction/fusion enrichies ML FastText-like + grammaire EN/FR/AR) ou "
            "pipeline100ml (embeddings Transformer BERT/XLM-R + pooling + grammaire XLM-R EN/FR/AR)."
        ),
    )
    parser.add_argument("--log-level", default="INFO", help="Niveau de log (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--only", choices=_STEP_CHOICES, help="N'executer qu'une seule etape (par nom).")
    parser.add_argument("--upto", choices=_STEP_CHOICES, help="Executer jusqu'a et incluant cette etape.")
    parser.add_argument("--start", choices=_STEP_CHOICES, help="Commencer a partir de cette etape.")
    parser.add_argument("--list-steps", action="store_true", help="Lister les etapes sans executer.")
    parser.add_argument(
        "--use-elasticsearch",
        action="store_true",
        default=_env_bool("USE_ELASTICSEARCH", False),
        help="Active Elasticsearch comme source texte pour classification/extraction et sync des resultats.",
    )
    parser.add_argument(
        "--es-url",
        default=os.environ.get("ES_URL", "http://localhost:9200"),
        help="URL Elasticsearch (ex: http://localhost:9200).",
    )
    parser.add_argument(
        "--es-index",
        default=os.environ.get("ES_INDEX", "dms_documents"),
        help="Nom d'index Elasticsearch utilise par le pipeline.",
    )
    parser.add_argument(
        "--es-nlp-level",
        choices=["off", "summary", "full"],
        default=os.environ.get("ES_NLP_LEVEL", "summary"),
        help=(
            "Niveau de synchronisation NLP vers Elasticsearch: "
            "off (desactive), summary (stats par document), "
            "full (summary + tokens detailles dans un index dedie)."
        ),
    )
    parser.add_argument(
        "--es-nlp-index",
        default=os.environ.get("ES_NLP_INDEX", "dms_nlp_tokens"),
        help="Nom de l'index Elasticsearch dedie aux tokens NLP (utilise en mode --es-nlp-level full).",
    )
    parser.add_argument(
        "--es-user",
        default=os.environ.get("ES_USER", ""),
        help="Utilisateur Elasticsearch (optionnel, sinon via .env).",
    )
    parser.add_argument(
        "--es-password",
        default=os.environ.get("ES_PASSWORD", ""),
        help="Mot de passe Elasticsearch (optionnel, sinon via .env).",
    )
    parser.add_argument(
        "--es-api-key",
        default=os.environ.get("ES_API_KEY", ""),
        help="API key Elasticsearch (optionnel, sinon via .env).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    args.only = _normalize_step_name(args.only)
    args.upto = _normalize_step_name(args.upto)
    args.start = _normalize_step_name(args.start)
    configure_logging(args.log_level)

    inputs = normalize_input(args.inputs) if args.inputs else []
    repo_root = Path(__file__).resolve().parent.parent
    postgres_status = ensure_postgres_bootstrap(repo_root, start_if_needed=False)
    pipeline_name = _normalize_pipeline_name(args.pipeline, "pipeline0ml")
    if pipeline_name == "pipeline100ml":
        orchestrator = Pipeline100MLOrchestrator(repo_root)
    elif pipeline_name == "pipeline50ml":
        orchestrator = Pipeline50MLOrchestrator(repo_root)
    else:
        orchestrator = Pipeline0MLOrchestrator(repo_root)
    logging.info("Pipeline selection: %s", pipeline_name)

    # Tee all prints to file + console
    log_path = repo_root / "outputgeneralterminal.txt"
    input_paths = []
    for raw in inputs:
        try:
            input_paths.append(Path(raw).expanduser().resolve())
        except Exception:
            continue
    if log_path.resolve() in input_paths:
        log_path = repo_root / "outputgeneralterminal.runtime.txt"
        logging.info(
            "outputgeneralterminal.txt est utilise comme document d'entree; logs rediriges vers %s",
            log_path,
        )

    log_file = log_path.open("w", encoding="utf-8")
    orig_print = builtins.print

    def tee_print(*pargs, **pkwargs):
        orig_print(*pargs, **pkwargs)
        sep = pkwargs.get("sep", " ")
        end = pkwargs.get("end", "\n")
        text = sep.join(str(a) for a in pargs) + end
        log_file.write(text)
        log_file.flush()

    builtins.print = tee_print

    if args.list_steps:
        for name in orchestrator.list_steps():
            print(name)
        builtins.print = orig_print
        log_file.close()
        return

    try:
        orchestrator.run(
            inputs,
            only=args.only,
            upto=args.upto,
            start=args.start,
            context_overrides={
                "USE_ELASTICSEARCH": bool(args.use_elasticsearch),
                "ES_URL": args.es_url,
                "ES_INDEX": args.es_index,
                "ES_NLP_LEVEL": args.es_nlp_level,
                "ES_NLP_INDEX": args.es_nlp_index,
                "ES_NLP_MAX_FULL_TOKENS": _env_int("ES_NLP_MAX_FULL_TOKENS", 200000),
                "ES_USER": (args.es_user or "").strip() or None,
                "ES_PASSWORD": args.es_password if args.es_password is not None else None,
                "ES_API_KEY": (args.es_api_key or "").strip() or None,
                "ES_AUTO_START": _env_bool("ES_AUTO_START", os.name == "nt"),
                "ES_START_COMMAND": _env_optional("ES_START_COMMAND"),
                "ES_START_COMMAND_POSIX": _env_optional("ES_START_COMMAND_POSIX"),
                "ES_START_COMMAND_WINDOWS": _env_optional("ES_START_COMMAND_WINDOWS"),
                "ES_START_COMMANDS": _env_optional("ES_START_COMMANDS"),
                "ES_START_PASSWORD": _env_optional("ES_START_PASSWORD"),
                "ES_AUTO_START_WAIT_SECONDS": _env_int("ES_AUTO_START_WAIT_SECONDS", 45),
                "ES_AUTO_START_LAUNCH_TIMEOUT": _env_int("ES_AUTO_START_LAUNCH_TIMEOUT", 20),
                "ML100_MODEL_NAME": _env_optional("ML100_MODEL_NAME") or "xlm-roberta-base",
                "ML100_MAX_LENGTH": _env_int("ML100_MAX_LENGTH", 256),
                "ML100_BATCH_SIZE": _env_int("ML100_BATCH_SIZE", 8),
                "ML100_HASH_FALLBACK_DIM": _env_int("ML100_HASH_FALLBACK_DIM", 384),
                "PIPELINE_PROFILE": pipeline_name,
                "POSTGRES_STATUS": postgres_status,
                "POSTGRES_ENABLED": bool(postgres_status.get("enabled")),
                "POSTGRES_READY": bool(postgres_status.get("ready")),
                "POSTGRES_SYNC_ENABLED": bool(postgres_status.get("sync_enabled")),
                "POSTGRES_HOST": postgres_status.get("host"),
                "POSTGRES_PORT": postgres_status.get("port"),
                "POSTGRES_USER": postgres_status.get("user"),
                "POSTGRES_DATABASE": postgres_status.get("database"),
                "POSTGRES_TABLES": postgres_status.get("tables_expected") or [],
                "POSTGRES_CONFIG_PATH": postgres_status.get("config_path"),
                "POSTGRES_SCHEMA_CONFIG_PATH": postgres_status.get("schema_config_path"),
            },
        )
    except Exception as exc:
        logging.exception("Echec du pipeline: %s", exc)
        raise
    finally:
        builtins.print = orig_print
        log_file.close()


if __name__ == "__main__":
    main()
