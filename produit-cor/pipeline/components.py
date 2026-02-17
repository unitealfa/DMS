from __future__ import annotations

import importlib.util
import logging
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from .settings import (
    COMPONENT_DIR,
    REPO_ROOT,
    Context,
    InputLike,
    change_dir,
    count_sentences,
    isolated_argv,
    normalize_input,
    safe_repr,
)


@dataclass
class Component:
    name: str
    script: Path

    def run(self, context: Context) -> Any:
        raise NotImplementedError

    def _execute_script(self, context: Context) -> Context:
        if not self.script.exists():
            raise FileNotFoundError(f"Script introuvable: {self.script}")
        logging.info("Execution du composant %s via %s", self.name, self.script)
        try:
            with change_dir(self.script.parent), isolated_argv([self.script.name]):
                result = runpy.run_path(
                    str(self.script),
                    run_name="__main__",
                    init_globals=context,
                )
                if result is not context:
                    context.clear()
                    context.update(result)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            raise RuntimeError(f"{self.name} a termine par sys.exit({code})") from exc
        except Exception:
            raise
        return context

    def _report(self, output: Any, summary: str) -> None:
        print(f"[Component: {self.name}]")
        print(f"Type: {type(output).__name__}")
        print(f"Summary: {summary}")
        print(f"Output: {safe_repr(output)}")
        print()


class PretraitementComponent(Component):
    def run(self, context: Context) -> Any:
        user_input = normalize_input(context.get("INPUT_FILE"))
        ctx = self._execute_script(context)

        default_input = ctx.get("INPUT_FILE")
        effective_input: Union[str, List[str], None] = user_input or default_input
        if effective_input is None:
            raise ValueError("Aucun fichier d'entree fourni pour le pretraitement.")

        ctx["INPUT_FILE"] = effective_input
        analyzer = ctx.get("analyze_many_two_states")
        if analyzer is None or not callable(analyzer):
            # Fallback: re-import the script as a module to fetch the function
            spec = importlib.util.spec_from_file_location(
                f"orchestrator_pretraitement_{self.script.stem.replace('-', '_')}",
                self.script,
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                analyzer = getattr(module, "analyze_many_two_states", None)
        if analyzer is None or not callable(analyzer):
            available = ", ".join(sorted(k for k in ctx.keys() if not k.startswith('__')))
            raise RuntimeError(
                f"Fonction analyze_many_two_states indisponible. Cles globals vues: {available}"
            )

        result = analyzer(effective_input)
        if not result:
            raise ValueError("pretraitement-de-docs n'a rien retourne.")

        ctx["PRETRAITEMENT_RESULT"] = result
        ctx["INPUT_FILE"] = [item["path"] for item in result if item.get("path")]

        summary = (
            f"{len(result)} fichiers | "
            f"text={sum(1 for d in result if d.get('content') == 'text')} | "
            f"image={sum(1 for d in result if d.get('content') == 'image_only')}"
        )
        self._report(result, summary)
        return result


class OCRPreprocessComponent(Component):
    def run(self, context: Context) -> Any:
        if "INPUT_FILE" not in context:
            raise ValueError("INPUT_FILE manquant avant OCR.")

        # Normaliser en chemins absolus pour eviter les soucis de cwd
        input_files: List[str] = []
        for p in context.get("INPUT_FILE") or []:
            path = Path(p)
            if not path.is_absolute():
                path = REPO_ROOT / path
            input_files.append(str(path))
        context["INPUT_FILE"] = input_files

        ctx = self._execute_script(context)

        text_files = ctx.get("TEXT_FILES") or []
        image_files = ctx.get("IMAGE_ONLY_FILES") or []
        docs = ctx.get("DOCS") or []

        # Si OCR n'a pas trouv d'images parce qu'elles ont t classes "text",
        # transformer les TEXT_FILES en DOCS vides pour que le pipeline continue.
        if not docs and image_files and not text_files:
            docs = []
        if not docs and not image_files and text_files:
            # inject minimal DOCS with source info to allow downstream extraction
            docs = [{"pages": [], "filename": Path(p).name, "source_files": [p], "page_count_total": 0} for p in text_files]

        if not text_files and not image_files and not docs:
            raise ValueError("si-image-pretraiter-sinonpass-le-doc n'a rien produit.")

        result = {
            "TEXT_FILES": text_files,
            "IMAGE_ONLY_FILES": image_files,
            "DOCS": docs,
        }
        ctx.update(result)
        ctx["PREPROCESS_RESULT"] = result
        summary = (
            f"text={len(result['TEXT_FILES'])}, "
            f"image={len(result['IMAGE_ONLY_FILES'])}, "
            f"docs_prepped={len(result['DOCS'])}"
        )
        self._report(result, summary)
        return result


class OutputTxtComponent(Component):
    def run(self, context: Context) -> Any:
        ctx = self._execute_script(context)
        final_docs = ctx.get("FINAL_DOCS")
        if not final_docs:
            raise ValueError("output-txt n'a retourne aucun FINAL_DOCS.")

        reader_with_name = ctx.get("_get_pdf_reader_with_name")
        if "_get_pdf_reader" not in ctx and callable(reader_with_name):
            ctx["_get_pdf_reader"] = lambda: reader_with_name()[0]

        summary = (
            f"{len(final_docs)} docs | "
            f"pages={sum(len(d.get('pages_text') or []) or 1 for d in final_docs)}"
        )
        self._report(final_docs, summary)
        return final_docs


class TokenisationLayoutComponent(Component):
    def run(self, context: Context) -> Any:
        ctx = self._execute_script(context)
        tok_docs = ctx.get("TOK_DOCS")
        selected = ctx.get("selected")

        output = tok_docs or selected
        if not output:
            raise ValueError("tokenisation-layout n'a produit aucun document tokenise.")

        total_sent = count_sentences(tok_docs or [])
        summary = f"{len(output)} docs | sentences={total_sent}"
        self._report(output, summary)
        return output


class GrammarComponent(Component):
    def run(self, context: Context) -> Any:
        # S'assure que les modules de langue sont trouvables avant excution
        lang_dir = COMPONENT_DIR / "atrribution-gramatical"
        if str(lang_dir) not in sys.path:
            sys.path.insert(0, str(lang_dir))

        ctx = self._execute_script(context)
        data = ctx.get("selected") or ctx.get("TOK_DOCS")
        if data is None:
            raise ValueError("atripusion-gramatical n'a pas recu de donnees.")

        langs = ["en", "fr"]
        if ctx.get("HAVE_AR"):
            langs.append("ar")

        summary = f"Langues executees: {', '.join(langs)} | docs={len(data) if isinstance(data, list) else 1}"
        self._report(data, summary)
        return data


class ClassificationComponent(Component):
    def run(self, context: Context) -> Any:
        ctx = self._execute_script(context)
        results = ctx.get("RESULTS")
        if not results:
            raise ValueError("clasification n'a retourne aucun resultat.")

        labels = ", ".join(f"{r.get('filename')}->{r.get('doc_type')}" for r in results)
        summary = f"{len(results)} docs classes | {labels}"
        self._report(results, summary)
        return results


