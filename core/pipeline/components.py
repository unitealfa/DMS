from __future__ import annotations

import importlib.util
import logging
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from .elasticsearch import (
    fetch_sources_for_ids,
    maybe_build_store,
    to_classification_docs,
    to_extraction_docs,
    update_classification_results,
    update_extraction_results,
)
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


def _doc_has_meaningful_text(doc: Dict[str, Any]) -> bool:
    if not isinstance(doc, dict):
        return False

    for key in ("text", "full_text"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return True

    for pg in doc.get("pages") or []:
        if not isinstance(pg, dict):
            continue
        for key in ("page_text", "ocr_text", "text"):
            value = pg.get(key)
            if isinstance(value, str) and value.strip():
                return True
        sent_items = pg.get("sentences_layout") or pg.get("sentences") or pg.get("chunks") or []
        for sent in sent_items:
            if isinstance(sent, dict):
                value = str(sent.get("text") or "")
            else:
                value = str(sent)
            if value.strip():
                return True
    return False


def _drop_empty_duplicate_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        paths = doc.get("paths")
        first_path = ""
        if isinstance(paths, list) and paths:
            first_path = str(paths[0])
        key = str(doc.get("filename") or first_path or doc.get("doc_id") or f"doc#{i}")
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(doc)

    out: List[Dict[str, Any]] = []
    for key in order:
        group = grouped[key]
        non_empty = [d for d in group if _doc_has_meaningful_text(d)]
        out.extend(non_empty if non_empty else group)
    return out


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

        missing_files = ctx.get("MISSING_FILES") or []
        ctx["PRETRAITEMENT_RESULT"] = result
        ctx["INPUT_FILE"] = [item["path"] for item in result if item.get("path")]
        ctx["MISSING_FILES"] = missing_files

        summary = (
            f"{len(result)} fichiers | "
            f"text={sum(1 for d in result if d.get('content') == 'text')} | "
            f"image={sum(1 for d in result if d.get('content') == 'image_only')} | "
            f"unsupported={sum(1 for d in result if d.get('content') == 'unsupported')} | "
            f"missing={len(missing_files)}"
        )
        output = {
            "PRETRAITEMENT_RESULT": result,
            "MISSING_FILES": missing_files,
        }
        self._report(output, summary)
        return output


class OCRPreprocessComponent(Component):
    def run(self, context: Context) -> Any:
        if "INPUT_FILE" not in context:
            raise ValueError("INPUT_FILE manquant avant OCR.")
        upstream_missing_files = list(context.get("MISSING_FILES") or [])

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
        unsupported_files = ctx.get("UNSUPPORTED_FILES") or []
        missing_files = []
        for item in upstream_missing_files + list(ctx.get("MISSING_FILES") or []):
            if item and item not in missing_files:
                missing_files.append(item)
        docs = ctx.get("DOCS") or []

        # Ne jamais injecter de DOCS artificiels quand les fichiers sont detectes en texte:
        # cela cree un faux document OCR vide dans output-txt.
        if not docs and image_files and not text_files:
            docs = []

        if not text_files and not image_files and not docs:
            raise ValueError("si-image-pretraiter-sinonpass-le-doc n'a rien produit.")

        result = {
            "TEXT_FILES": text_files,
            "IMAGE_ONLY_FILES": image_files,
            "UNSUPPORTED_FILES": unsupported_files,
            "MISSING_FILES": missing_files,
            "DOCS": docs,
        }
        ctx.update(result)
        ctx["PREPROCESS_RESULT"] = result
        summary = (
            f"text={len(result['TEXT_FILES'])}, "
            f"image={len(result['IMAGE_ONLY_FILES'])}, "
            f"unsupported={len(result['UNSUPPORTED_FILES'])}, "
            f"missing={len(result['MISSING_FILES'])}, "
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

        def _doc_pages_count(row: Dict[str, Any]) -> int:
            raw = row.get("page_count_total")
            if isinstance(raw, int) and raw >= 0:
                return raw
            return len(row.get("pages_text") or [])

        summary = f"{len(final_docs)} docs | pages={sum(_doc_pages_count(d) for d in final_docs)}"
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

        if isinstance(output, list):
            filtered = _drop_empty_duplicate_docs(output)
            removed = len(output) - len(filtered)
            if removed > 0:
                logging.info("Tokenisation: %d document(s) vide(s) en doublon ignores.", removed)
            if tok_docs is not None:
                ctx["TOK_DOCS"] = filtered
            if selected is not None:
                ctx["selected"] = filtered
            output = filtered

        total_sent = count_sentences(output or [])
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

        # Audit explicite du composant grammaire utilise.
        ctx["NLP_GRAMMAR_COMPONENT_NAME"] = self.name
        ctx["NLP_GRAMMAR_COMPONENT_SCRIPT"] = str(self.script)
        if not ctx.get("NLP_GRAMMAR_BACKEND"):
            script_name = self.script.name.lower()
            if "attribution-gramatical-100ml-xlmr" in script_name:
                ctx["NLP_GRAMMAR_BACKEND"] = "xlmr"
            else:
                ctx["NLP_GRAMMAR_BACKEND"] = "legacy-3files"

        langs = ["en", "fr"]
        if ctx.get("HAVE_AR"):
            langs.append("ar")

        backend = str(ctx.get("NLP_GRAMMAR_BACKEND") or "unknown")
        pos_method = str(ctx.get("NLP_POS_METHOD") or "").strip()
        pos_refined = int(ctx.get("NLP_POS_REFINED_COUNT") or 0)
        pos_total = int(ctx.get("NLP_POS_TOTAL") or 0)
        summary = (
            f"Langues executees: {', '.join(langs)} | docs={len(data) if isinstance(data, list) else 1} | "
            f"backend={backend} | component={self.name}"
        )
        if pos_method:
            summary += f" | pos={pos_method}"
        if pos_total > 0:
            summary += f" | pos_refined={pos_refined}/{pos_total}"
        self._report(data, summary)
        return data


class TableExtractionComponent(Component):
    def run(self, context: Context) -> Any:
        ctx = self._execute_script(context)
        rows = ctx.get("TABLE_EXTRACTIONS")
        if not isinstance(rows, list):
            rows = ctx.get("TABLE_EXTRACTIONS_50ML") or ctx.get("TABLE_EXTRACTIONS_100ML") or []
        if not isinstance(rows, list):
            rows = []

        docs = len(rows)
        tables = 0
        line_items = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            tables += int(row.get("tables_count") or 0)
            line_items += int(row.get("rows_total") or 0)

        summary = f"docs={docs} | tables={tables} | line_items={line_items}"
        self._report(rows, summary)
        return rows


class TotalsVerificationComponent(Component):
    def run(self, context: Context) -> Any:
        ctx = self._execute_script(context)
        rows = ctx.get("TOTALS_VERIFICATION") or []
        if not isinstance(rows, list):
            rows = []

        docs = len(rows)
        ok = sum(1 for row in rows if isinstance(row, dict) and str(row.get("verification_status")) == "ok")
        partial_ok = sum(1 for row in rows if isinstance(row, dict) and str(row.get("verification_status")) == "partial_ok")
        mismatch = sum(1 for row in rows if isinstance(row, dict) and str(row.get("verification_status")) == "mismatch")
        missing = sum(1 for row in rows if isinstance(row, dict) and str(row.get("verification_status")) == "not_enough_data")
        summary = (
            f"docs={docs} | ok={ok} | partial_ok={partial_ok} | "
            f"mismatch={mismatch} | missing={missing}"
        )
        self._report(rows, summary)
        return rows


class VisualMarksDetectionComponent(Component):
    def run(self, context: Context) -> Any:
        ctx = self._execute_script(context)
        rows = ctx.get("VISUAL_MARKS_DETECTIONS_100ML")
        if not isinstance(rows, list):
            rows = ctx.get("VISUAL_MARKS_DETECTIONS") or []
        if not isinstance(rows, list):
            rows = []

        docs = len(rows)
        detections = 0
        signature = 0
        stamp = 0
        barcode = 0
        qrcode = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            detections += int(row.get("detections_count") or len(row.get("detections") or []))
            signature += int(row.get("counts", {}).get("signature") or 0) if isinstance(row.get("counts"), dict) else 0
            stamp += int(row.get("counts", {}).get("stamp") or 0) if isinstance(row.get("counts"), dict) else 0
            barcode += int(row.get("counts", {}).get("barcode") or 0) if isinstance(row.get("counts"), dict) else 0
            qrcode += int(row.get("counts", {}).get("qrcode") or 0) if isinstance(row.get("counts"), dict) else 0

        summary = (
            f"docs={docs} | detections={detections} | "
            f"signature={signature} | stamp={stamp} | barcode={barcode} | qrcode={qrcode}"
        )
        self._report(rows, summary)
        return rows


class InterDocLinkingComponent(Component):
    def run(self, context: Context) -> Any:
        ctx = self._execute_script(context)
        analysis = ctx.get("INTERDOC_ANALYSIS") if isinstance(ctx.get("INTERDOC_ANALYSIS"), dict) else {}
        links = ctx.get("INTERDOC_LINKS")
        if not isinstance(links, list):
            links = analysis.get("links") if isinstance(analysis.get("links"), list) else []

        output = {
            "method": analysis.get("method"),
            "documents_analyzed": int(analysis.get("documents_analyzed") or 0),
            "pairs_evaluated": int(analysis.get("pairs_evaluated") or 0),
            "sentence_pairs_scored": int(analysis.get("sentence_pairs_scored") or 0),
            "chunk_pairs_scored": int(analysis.get("chunk_pairs_scored") or 0),
            "links_count": int(analysis.get("links_count") or len(links)),
            "vector_profile": analysis.get("vector_profile"),
            "vector_links_count": int(analysis.get("vector_links_count") or 0),
        }
        summary = (
            f"docs={output['documents_analyzed']} | pairs={output['pairs_evaluated']} | "
            f"links={output['links_count']} | sentence_pairs={output['sentence_pairs_scored']} | "
            f"chunk_pairs={output['chunk_pairs_scored']}"
        )
        if output.get("vector_profile"):
            summary += (
                f" | vector_profile={output['vector_profile']} | "
                f"vector_links={output['vector_links_count']}"
            )
        self._report(output, summary)
        return output


class ElasticsearchComponent(Component):
    def run(self, context: Context) -> Any:
        ctx = self._execute_script(context)
        enabled = bool(ctx.get("USE_ELASTICSEARCH"))
        available = bool(ctx.get("ES_AVAILABLE"))
        doc_ids = ctx.get("ES_DOC_IDS") or []
        cls_docs = ctx.get("ES_CLASSIFICATION_DOCS") or []
        ext_docs = ctx.get("ES_EXTRACTION_DOCS") or []
        class_results = ctx.get("RESULTS") if isinstance(ctx.get("RESULTS"), list) else []
        class_synced = int(ctx.get("ES_CLASSIFICATION_SYNCED") or 0)
        nlp_sync = ctx.get("ES_NLP_SYNC") if isinstance(ctx.get("ES_NLP_SYNC"), dict) else {}

        output = {
            "enabled": enabled,
            "available": available,
            "es_url": ctx.get("ES_URL"),
            "es_index": ctx.get("ES_INDEX"),
            "doc_ids": len(doc_ids),
            "classification_input": len(class_results),
            "classification_synced": class_synced,
            "classification_docs": len(cls_docs),
            "extraction_docs": len(ext_docs),
            "nlp_level": nlp_sync.get("level"),
            "nlp_docs_synced": int(nlp_sync.get("docs_synced") or 0),
            "nlp_tokens_indexed": int(nlp_sync.get("tokens_indexed") or 0),
            "nlp_token_errors": int(nlp_sync.get("token_index_errors") or 0),
            "nlp_unresolved_rows": int(nlp_sync.get("unresolved_rows") or 0),
            "nlp_tokens_index": nlp_sync.get("tokens_index"),
        }
        summary = (
            f"enabled={enabled} | available={available} | indexed={len(doc_ids)} | "
            f"cls_input={len(class_results)} | cls_synced={class_synced} | "
            f"cls_docs={len(cls_docs)} | ext_docs={len(ext_docs)} | "
            f"nlp={nlp_sync.get('level')} docs={int(nlp_sync.get('docs_synced') or 0)} "
            f"tokens={int(nlp_sync.get('tokens_indexed') or 0)} "
            f"errors={int(nlp_sync.get('token_index_errors') or 0)}"
        )
        self._report(output, summary)
        return output


class ClassificationComponent(Component):
    def run(self, context: Context) -> Any:
        store = None
        try:
            store = maybe_build_store(context)
            if store is not None:
                if not context.get("ES_CLASSIFICATION_DOCS"):
                    es_ids = [str(x) for x in (context.get("ES_DOC_IDS") or []) if x]
                    if es_ids:
                        es_sources = fetch_sources_for_ids(store, es_ids)
                        es_docs = to_classification_docs(es_sources)
                        if es_docs:
                            context["ES_CLASSIFICATION_DOCS"] = es_docs
                            logging.info(
                                "Classification: fallback Elasticsearch active (%d docs).",
                                len(es_docs),
                            )
        except Exception as exc:
            logging.warning("Elasticsearch pre-classification desactive: %s", exc)
            store = None

        ctx = self._execute_script(context)
        results = ctx.get("RESULTS")
        if not results:
            raise ValueError("clasification n'a retourne aucun resultat.")

        if store is not None:
            es_ids = [str(x) for x in (ctx.get("ES_DOC_IDS") or []) if x]
            if es_ids:
                try:
                    synced = update_classification_results(store, results)
                    ctx["ES_CLASSIFICATION_SYNCED"] = synced
                    logging.info("Classification synchronisee vers Elasticsearch: %d document(s).", synced)
                except Exception as exc:
                    logging.warning("Sync classification vers Elasticsearch echouee: %s", exc)
            else:
                ctx["ES_CLASSIFICATION_SYNCED"] = int(ctx.get("ES_CLASSIFICATION_SYNCED") or 0)
                logging.info(
                    "Classification calculee mais sync Elasticsearch differee: indexation docs non executee (etape elasticsearch plus tard)."
                )

        labels = ", ".join(f"{r.get('filename')}->{r.get('doc_type')}" for r in results)
        summary = f"{len(results)} docs classes | {labels}"
        self._report(results, summary)
        return results


class RuleExtractionComponent(Component):
    def run(self, context: Context) -> Any:
        store = None
        try:
            store = maybe_build_store(context)
            if store is not None:
                if not context.get("ES_EXTRACTION_DOCS"):
                    es_ids: List[str] = [str(x) for x in (context.get("ES_DOC_IDS") or []) if x]
                    if es_ids:
                        es_sources = fetch_sources_for_ids(store, es_ids)
                        es_docs = to_extraction_docs(es_sources)
                        if es_docs:
                            context["ES_EXTRACTION_DOCS"] = es_docs
                            logging.info(
                                "Extraction-regles: fallback Elasticsearch active (%d docs).",
                                len(es_docs),
                            )
        except Exception as exc:
            logging.warning("Elasticsearch pre-extraction desactive: %s", exc)
            store = None

        ctx = self._execute_script(context)
        extractions = ctx.get("EXTRACTIONS")
        if extractions is None:
            raise ValueError("extraction-regles n'a retourne aucun resultat.")

        if store is not None:
            try:
                synced = update_extraction_results(store, extractions)
                ctx["ES_EXTRACTION_SYNCED"] = synced
                logging.info("Extraction synchronisee vers Elasticsearch: %d document(s).", synced)
            except Exception as exc:
                logging.warning("Sync extraction vers Elasticsearch echouee: %s", exc)

        total_fields = 0
        labels = []
        if isinstance(extractions, list):
            total_fields = sum(len(d.get("fields") or {}) for d in extractions)
            labels = [f"{d.get('filename')}->{d.get('doc_type')}" for d in extractions]

        summary = f"{len(extractions) if isinstance(extractions, list) else 0} docs | fields={total_fields}"
        if labels:
            summary += f" | {', '.join(labels[:4])}"
            if len(labels) > 4:
                summary += " ..."

        self._report(extractions, summary)
        return extractions


class FusionResultComponent(Component):
    """Fusionne les sorties en un JSON final (voir component/fusion_resultats.py)."""

    def run(self, context: Context) -> Any:
        if not self.script.exists():
            output = {
                "path": None,
                "source": "disabled",
                "docs": 0,
                "es_synced": 0,
                "skipped": True,
            }
            self._report(output, "fusion debug absent -> ignore")
            return output

        try:
            ctx = self._execute_script(context)
        except Exception as exc:
            logging.warning("fusion-resultats (debug) ignore apres erreur: %s", exc)
            output = {
                "path": None,
                "source": "disabled",
                "docs": 0,
                "es_synced": 0,
                "skipped": True,
                "error": str(exc),
            }
            self._report(output, "fusion debug en erreur -> ignore")
            return output

        fusion = ctx.get("FUSION_RESULT") or "fusion_output.json"
        source = ctx.get("FUSION_SOURCE") or "local-context"
        doc_count = len(ctx.get("FUSION_PAYLOADS") or [])
        es_synced = int(ctx.get("ES_FUSION_SYNCED") or 0)
        output = {
            "path": fusion,
            "source": source,
            "docs": doc_count,
            "es_synced": es_synced,
            "skipped": False,
        }
        summary = f"fusion -> {fusion} | source={source} | docs={doc_count} | es_synced={es_synced}"
        self._report(output, summary)
        return output


class PostgresSyncComponent(Component):
    """Synchronise le `fusion_output.json` final vers PostgreSQL."""

    def run(self, context: Context) -> Any:
        ctx = self._execute_script(context)
        output = ctx.get("POSTGRES_SYNC")
        if not isinstance(output, dict):
            raise ValueError("postgres-sync n'a retourne aucun POSTGRES_SYNC exploitable.")

        summary = (
            f"ready={1 if output.get('ready') else 0} | "
            f"db={output.get('database')} | "
            f"run={output.get('run_id')} | "
            f"logs={int(output.get('log_entries_created') or 0)} | "
            f"docs={int(output.get('documents_total') or 0)}"
        )
        if output.get("skipped"):
            summary += f" | skipped={output.get('skipped')}"
        if output.get("error"):
            summary += f" | error={output.get('error')}"
        self._report(output, summary)
        return output
