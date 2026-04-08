from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .components import (
    APIOutputComponent,
    ClassificationComponent,
    Component,
    ElasticsearchComponent,
    FusionResultComponent,
    GrammarComponent,
    InterDocLinkingComponent,
    OCRPreprocessComponent,
    OutputTxtComponent,
    PretraitementComponent,
    RuleExtractionComponent,
    TableExtractionComponent,
    TokenisationLayoutComponent,
    TotalsVerificationComponent,
    VisualMarksDetectionComponent,
)
from .runtime_state import publish_pipeline_completed, publish_pipeline_failed, publish_pipeline_started
from .settings import COMPONENT_DIR, Context, InputLike, normalize_input


class BasePipelineOrchestrator:
    code = "pipeline"
    aliases: tuple[str, ...] = ()
    label = "Pipeline"
    description = ""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.components = self.build_components()

    def build_components(self) -> List[Any]:
        raise NotImplementedError

    def list_steps(self) -> List[str]:
        return [c.name for c in self.components]

    def _select_components(self, only: Optional[str], upto: Optional[str], start: Optional[str]) -> List[Any]:
        comps = self.components
        steps = self.list_steps()
        if start:
            if start not in steps:
                raise ValueError(f"Etape inconnue (start): {start}")
            comps = comps[steps.index(start) :]
        if upto:
            if upto not in steps:
                raise ValueError(f"Etape inconnue (upto): {upto}")
            upto_index = steps.index(upto)
            comps = [c for c in comps if steps.index(c.name) <= upto_index]
        if only:
            if only not in steps:
                raise ValueError(f"Etape inconnue (only): {only}")
            comps = [c for c in self.components if c.name == only]
        return comps

    def run(
        self,
        input_files: InputLike,
        only: Optional[str] = None,
        upto: Optional[str] = None,
        start: Optional[str] = None,
        context_overrides: Optional[Context] = None,
    ) -> Context:
        context: Context = {"INPUT_FILE": normalize_input(input_files)}
        if context_overrides:
            context.update(context_overrides)
        context["PIPELINE_PROFILE"] = str(type(self).code)
        selected = self._select_components(only, upto, start)
        context["PIPELINE_STEPS"] = [c.name for c in selected]
        publish_pipeline_started(context)
        try:
            for comp in selected:
                output: Any = comp.run(context)
                if output is None:
                    raise RuntimeError(f"{comp.name} a retourne None.")
        except Exception as exc:
            publish_pipeline_failed(context, error=exc)
            raise
        publish_pipeline_completed(context)
        return context


def _pipeline_sort_key(cls: Type[BasePipelineOrchestrator]) -> tuple[str, int, str]:
    try:
        source_file = inspect.getsourcefile(cls) or ""
        _, line_no = inspect.getsourcelines(cls)
        return (source_file, int(line_no or 0), cls.__name__)
    except Exception:
        return (cls.__module__, 10**9, cls.__name__)


def pipeline_orchestrator_classes() -> List[Type[BasePipelineOrchestrator]]:
    seen: set[type[BasePipelineOrchestrator]] = set()
    ordered: List[Type[BasePipelineOrchestrator]] = []
    stack = list(BasePipelineOrchestrator.__subclasses__())
    while stack:
        cls = stack.pop(0)
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if getattr(cls, "code", None):
            ordered.append(cls)
    ordered.sort(key=_pipeline_sort_key)
    return ordered


def pipeline_registry() -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for cls in pipeline_orchestrator_classes():
        code = str(getattr(cls, "code", "") or "").strip()
        if not code:
            continue
        aliases = tuple(str(alias).strip() for alias in (getattr(cls, "aliases", ()) or ()) if str(alias).strip())
        registry[code] = {
            "code": code,
            "aliases": aliases,
            "label": str(getattr(cls, "label", "") or code),
            "description": str(getattr(cls, "description", "") or ""),
            "class": cls,
        }
    return registry


def available_pipeline_codes() -> List[str]:
    return list(pipeline_registry().keys())


def available_pipeline_choices(default_code: str | None = None) -> List[str]:
    codes = available_pipeline_codes()
    if not codes:
        return ["default"]
    normalized_default = normalize_pipeline_name(default_code, codes[0])
    if normalized_default not in codes:
        normalized_default = codes[0]
    return ["default", *codes]


def pipeline_definition(profile: str | None, default: str | None = None) -> Dict[str, Any]:
    registry = pipeline_registry()
    codes = list(registry.keys())
    if not registry:
        raise RuntimeError("Aucun pipeline enregistre.")
    resolved = normalize_pipeline_name(profile, default or codes[0])
    return registry.get(resolved) or registry[codes[0]]


def normalize_pipeline_name(raw: str | None, default: str | None = None) -> str:
    registry = pipeline_registry()
    codes = list(registry.keys())
    if not codes:
        return str(default or "pipeline")

    base_default = str(default or "").strip() or codes[0]
    if base_default not in registry:
        alias_probe = str(base_default).strip().lower()
        resolved_default = None
        for code, meta in registry.items():
            if alias_probe == code.lower() or alias_probe in {alias.lower() for alias in meta.get("aliases", ())}:
                resolved_default = code
                break
        base_default = resolved_default or codes[0]

    value = str(raw or "").strip().lower()
    if not value:
        return base_default
    if value == "default":
        return base_default

    for code, meta in registry.items():
        if value == code.lower():
            return code
        for alias in meta.get("aliases", ()) or ():
            if value == str(alias).strip().lower():
                return code
    return base_default


def create_pipeline_orchestrator(profile: str | None, base_dir: Path, default: str | None = None) -> BasePipelineOrchestrator:
    meta = pipeline_definition(profile, default=default)
    cls = meta["class"]
    return cls(base_dir)


class Pipeline0MLOrchestrator(BasePipelineOrchestrator):
    code = "pipeline0ml"
    aliases = ("0ml",)
    label = "Pipeline 0ML"
    description = "Baseline non-ML routing with classic tokenisation, grammar, rule extraction and totals verification."

    def build_components(self) -> List[Any]:
        return [
            PretraitementComponent("pretraitement-de-docs", COMPONENT_DIR / "pretraitement-de-docs.py"),
            OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", COMPONENT_DIR / "si-image-pretraiter-sinonpass-le-doc.py"),
            OutputTxtComponent("output-txt", COMPONENT_DIR / "output-txt.py"),
            ClassificationComponent("clasification", COMPONENT_DIR / "clasification.py"),
            TokenisationLayoutComponent("tokenisation-layout", COMPONENT_DIR / "tokenisation_layout" / "tokenisation-layout.py"),
            GrammarComponent("atripusion-gramatical", COMPONENT_DIR / "atrribution-gramatical" / "atripusion-gramatical-en-utilisant-les3ficherla.py"),
            TableExtractionComponent("table-extraction", COMPONENT_DIR / "table_extraction" / "table-extraction.py"),
            TotalsVerificationComponent("verification-totaux", COMPONENT_DIR / "verification-totaux.py"),
            InterDocLinkingComponent("liaison-inter-docs", COMPONENT_DIR / "liaison-inter-docs.py"),
            ElasticsearchComponent("elasticsearch", COMPONENT_DIR / "elasticsearch.py"),
            RuleExtractionComponent("extraction-regles", COMPONENT_DIR / "extraction" / "extraction-regles.py"),
            FusionResultComponent("fusion-resultats", COMPONENT_DIR / "fusion_resultats.py"),
            APIOutputComponent("api-output", COMPONENT_DIR / "api-output.py"),
        ]


class Pipeline50MLOrchestrator(BasePipelineOrchestrator):
    code = "pipeline50ml"
    aliases = ("50ml",)
    label = "Pipeline 50ML"
    description = "Hybrid ML retrieval pipeline with 50ML tokenisation and extraction components plus grammar refinement."

    def build_components(self) -> List[Any]:
        return [
            PretraitementComponent("pretraitement-de-docs", COMPONENT_DIR / "pretraitement-de-docs.py"),
            OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", COMPONENT_DIR / "si-image-pretraiter-sinonpass-le-doc.py"),
            OutputTxtComponent("output-txt", COMPONENT_DIR / "output-txt.py"),
            ClassificationComponent("clasification", COMPONENT_DIR / "clasification.py"),
            TokenisationLayoutComponent("tokenisation-layout", COMPONENT_DIR / "tokenisation_layout" / "tokenisation-layout-50ml.py"),
            GrammarComponent("atripusion-gramatical", COMPONENT_DIR / "atrribution-gramatical" / "atripusion-gramatical-en-utilisant-les3ficherla.py"),
            TableExtractionComponent("table-extraction", COMPONENT_DIR / "table_extraction" / "table-extraction.py"),
            TotalsVerificationComponent("verification-totaux", COMPONENT_DIR / "verification-totaux.py"),
            InterDocLinkingComponent("liaison-inter-docs", COMPONENT_DIR / "liaison-inter-docs.py"),
            ElasticsearchComponent("elasticsearch", COMPONENT_DIR / "elasticsearch.py"),
            RuleExtractionComponent("extraction-regles", COMPONENT_DIR / "extraction" / "extraction-regles-50ml.py"),
            FusionResultComponent("fusion-resultats", COMPONENT_DIR / "fusion_resultats.py"),
            APIOutputComponent("api-output", COMPONENT_DIR / "api-output.py"),
        ]


class Pipeline100MLOrchestrator(BasePipelineOrchestrator):
    code = "pipeline100ml"
    aliases = ("100ml",)
    label = "Pipeline 100ML"
    description = "Transformer-grade pipeline with 100ML tokenisation, XLM-R grammar and visual marks detection."

    def build_components(self) -> List[Any]:
        return [
            PretraitementComponent("pretraitement-de-docs", COMPONENT_DIR / "pretraitement-de-docs.py"),
            OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", COMPONENT_DIR / "si-image-pretraiter-sinonpass-le-doc.py"),
            OutputTxtComponent("output-txt", COMPONENT_DIR / "output-txt.py"),
            ClassificationComponent("clasification", COMPONENT_DIR / "clasification.py"),
            TokenisationLayoutComponent("tokenisation-layout", COMPONENT_DIR / "tokenisation_layout" / "tokenisation-layout-100ml.py"),
            GrammarComponent("atripusion-gramatical", COMPONENT_DIR / "atrribution-gramatical" / "attribution-gramatical-100ml-xlmr.py"),
            TableExtractionComponent("table-extraction", COMPONENT_DIR / "table_extraction" / "table-extraction.py"),
            TotalsVerificationComponent("verification-totaux", COMPONENT_DIR / "verification-totaux.py"),
            VisualMarksDetectionComponent("detection-signature-chachet-codebarr", COMPONENT_DIR / "detection-signature-chachet-codebarr.py"),
            InterDocLinkingComponent("liaison-inter-docs", COMPONENT_DIR / "liaison-inter-docs.py"),
            ElasticsearchComponent("elasticsearch", COMPONENT_DIR / "elasticsearch.py"),
            RuleExtractionComponent("extraction-regles", COMPONENT_DIR / "extraction" / "extraction-regles-100ml.py"),
            FusionResultComponent("fusion-resultats", COMPONENT_DIR / "fusion_resultats.py"),
            APIOutputComponent("api-output", COMPONENT_DIR / "api-output.py"),
        ]
