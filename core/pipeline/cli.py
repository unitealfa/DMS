from __future__ import annotations

import argparse
import logging
import builtins
from pathlib import Path

from .orchestrator import PipelineOrchestrator
from .settings import configure_logging, normalize_input


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrateur du pipeline documentaire.")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Chemins des fichiers a traiter (separes par des virgules ou des espaces).",
    )
    parser.add_argument("--log-level", default="INFO", help="Niveau de log (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--only", choices=[
        "pretraitement-de-docs",
        "si-image-pretraiter-sinonpass-le-doc",
        "output-txt",
        "tokenisation-layout",
        "atripusion-gramatical-en-utilisant-les3ficherla",
        "clasification",
        "extraction-regles",
        "fusion-resultats",
    ], help="N'executer qu'une seule etape (par nom).")
    parser.add_argument("--upto", choices=[
        "pretraitement-de-docs",
        "si-image-pretraiter-sinonpass-le-doc",
        "output-txt",
        "tokenisation-layout",
        "atripusion-gramatical-en-utilisant-les3ficherla",
        "clasification",
        "extraction-regles",
        "fusion-resultats",
    ], help="Executer jusqu'a et incluant cette etape.")
    parser.add_argument("--start", choices=[
        "pretraitement-de-docs",
        "si-image-pretraiter-sinonpass-le-doc",
        "output-txt",
        "tokenisation-layout",
        "atripusion-gramatical-en-utilisant-les3ficherla",
        "clasification",
        "extraction-regles",
        "fusion-resultats",
    ], help="Commencer a partir de cette etape.")
    parser.add_argument("--list-steps", action="store_true", help="Lister les etapes sans executer.")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    configure_logging(args.log_level)

    inputs = normalize_input(args.inputs) if args.inputs else []
    repo_root = Path(__file__).resolve().parent.parent
    orchestrator = PipelineOrchestrator(repo_root)

    # Tee all prints to file + console
    log_file = (repo_root / "outputgeneralterminal.txt").open("w", encoding="utf-8")
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
        orchestrator.run(inputs, only=args.only, upto=args.upto, start=args.start)
    except Exception as exc:
        logging.exception("Echec du pipeline: %s", exc)
        raise
    finally:
        builtins.print = orig_print
        log_file.close()


if __name__ == "__main__":
    main()
