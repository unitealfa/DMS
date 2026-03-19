from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from .components import (
    ClassificationComponent,
    ElasticsearchComponent,
    InterDocLinkingComponent,
    RuleExtractionComponent,
    GrammarComponent,
    OCRPreprocessComponent,
    OutputTxtComponent,
    PretraitementComponent,
    TableExtractionComponent,
    TokenisationLayoutComponent,
    FusionResultComponent,
)
from .settings import COMPONENT_DIR, Context, InputLike, normalize_input


class PipelineOrchestrator:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.components = [
            PretraitementComponent("pretraitement-de-docs", COMPONENT_DIR / "pretraitement-de-docs.py"),
            OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", COMPONENT_DIR / "si-image-pretraiter-sinonpass-le-doc.py"),
            OutputTxtComponent("output-txt", COMPONENT_DIR / "output-txt.py"),
            ClassificationComponent("clasification", COMPONENT_DIR / "clasification.py"),
            TokenisationLayoutComponent("tokenisation-layout", COMPONENT_DIR / "tokenisation_layout" / "tokenisation-layout.py"),
            GrammarComponent("atripusion-gramatical", COMPONENT_DIR / "atrribution-gramatical" / "atripusion-gramatical-en-utilisant-les3ficherla.py"),
            InterDocLinkingComponent("liaison-inter-docs", COMPONENT_DIR / "liaison-inter-docs.py"),
            ElasticsearchComponent("elasticsearch", COMPONENT_DIR / "elasticsearch.py"),
            RuleExtractionComponent("extraction-regles", COMPONENT_DIR / "extraction" / "extraction-regles.py"),
            FusionResultComponent("fusion-resultats", COMPONENT_DIR / "fusion_resultats.py"),
        ]

    def list_steps(self) -> List[str]:
        return [c.name for c in self.components]

    def _select_components(self, only: Optional[str], upto: Optional[str], start: Optional[str]) -> List[Any]:
        comps = self.components
        if start:
            if start not in self.list_steps():
                raise ValueError(f"Etape inconnue (start): {start}")
            idx = self.list_steps().index(start)
            comps = comps[idx:]
        if upto:
            if upto not in self.list_steps():
                raise ValueError(f"Etape inconnue (upto): {upto}")
            idx = self.list_steps().index(upto)
            comps = [c for c in comps if self.list_steps().index(c.name) <= idx]
        if only:
            if only not in self.list_steps():
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
        context["PIPELINE_PROFILE"] = str(context.get("PIPELINE_PROFILE") or "default")
        selected = self._select_components(only, upto, start)
        context["PIPELINE_STEPS"] = [c.name for c in selected]
        for comp in selected:
            output: Any = comp.run(context)
            if output is None:
                raise RuntimeError(f"{comp.name} a retourne None.")
        return context


class Pipeline50MLOrchestrator:
    """
    Variante de pipeline orientee ML retrieval:
    - conserve l'etape grammaire pour raffiner les sorties NLP/topics
    - remplace tokenisation/extraction/fusion par versions 50ml
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.components = [
            PretraitementComponent("pretraitement-de-docs", COMPONENT_DIR / "pretraitement-de-docs.py"),
            OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", COMPONENT_DIR / "si-image-pretraiter-sinonpass-le-doc.py"),
            OutputTxtComponent("output-txt", COMPONENT_DIR / "output-txt.py"),
            ClassificationComponent("clasification", COMPONENT_DIR / "clasification.py"),
            TokenisationLayoutComponent("tokenisation-layout", COMPONENT_DIR / "tokenisation_layout" / "tokenisation-layout-50ml.py"),
            GrammarComponent("atripusion-gramatical", COMPONENT_DIR / "atrribution-gramatical" / "atripusion-gramatical-en-utilisant-les3ficherla.py"),
            TableExtractionComponent("table-extraction", COMPONENT_DIR / "table_extraction" / "table-extraction.py"),
            InterDocLinkingComponent("liaison-inter-docs", COMPONENT_DIR / "liaison-inter-docs.py"),
            ElasticsearchComponent("elasticsearch", COMPONENT_DIR / "elasticsearch.py"),
            RuleExtractionComponent("extraction-regles", COMPONENT_DIR / "extraction" / "extraction-regles-50ml.py"),
            FusionResultComponent("fusion-resultats", COMPONENT_DIR / "fusion_resultats.py"),
        ]

    def list_steps(self) -> List[str]:
        return [c.name for c in self.components]

    def _select_components(self, only: Optional[str], upto: Optional[str], start: Optional[str]) -> List[Any]:
        comps = self.components
        if start:
            if start not in self.list_steps():
                raise ValueError(f"Etape inconnue (start): {start}")
            idx = self.list_steps().index(start)
            comps = comps[idx:]
        if upto:
            if upto not in self.list_steps():
                raise ValueError(f"Etape inconnue (upto): {upto}")
            idx = self.list_steps().index(upto)
            comps = [c for c in comps if self.list_steps().index(c.name) <= idx]
        if only:
            if only not in self.list_steps():
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
        context["PIPELINE_PROFILE"] = "pipeline50ml"
        selected = self._select_components(only, upto, start)
        context["PIPELINE_STEPS"] = [c.name for c in selected]
        for comp in selected:
            output: Any = comp.run(context)
            if output is None:
                raise RuntimeError(f"{comp.name} a retourne None.")
        return context


class Pipeline100MLOrchestrator:
    """
    Variante de pipeline orientee embeddings Transformer:
    - conserve l'etape grammaire pour raffiner les sorties NLP/topics
    - remplace tokenisation/extraction/fusion par versions 100ml
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.components = [
            PretraitementComponent("pretraitement-de-docs", COMPONENT_DIR / "pretraitement-de-docs.py"),
            OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", COMPONENT_DIR / "si-image-pretraiter-sinonpass-le-doc.py"),
            OutputTxtComponent("output-txt", COMPONENT_DIR / "output-txt.py"),
            ClassificationComponent("clasification", COMPONENT_DIR / "clasification.py"),
            TokenisationLayoutComponent("tokenisation-layout", COMPONENT_DIR / "tokenisation_layout" / "tokenisation-layout-100ml.py"),
            GrammarComponent("atripusion-gramatical", COMPONENT_DIR / "atrribution-gramatical" / "attribution-gramatical-100ml-xlmr.py"),
            TableExtractionComponent("table-extraction", COMPONENT_DIR / "table_extraction" / "table-extraction.py"),
            InterDocLinkingComponent("liaison-inter-docs", COMPONENT_DIR / "liaison-inter-docs.py"),
            ElasticsearchComponent("elasticsearch", COMPONENT_DIR / "elasticsearch.py"),
            RuleExtractionComponent("extraction-regles", COMPONENT_DIR / "extraction" / "extraction-regles-100ml.py"),
            FusionResultComponent("fusion-resultats", COMPONENT_DIR / "fusion_resultats.py"),
        ]

    def list_steps(self) -> List[str]:
        return [c.name for c in self.components]

    def _select_components(self, only: Optional[str], upto: Optional[str], start: Optional[str]) -> List[Any]:
        comps = self.components
        if start:
            if start not in self.list_steps():
                raise ValueError(f"Etape inconnue (start): {start}")
            idx = self.list_steps().index(start)
            comps = comps[idx:]
        if upto:
            if upto not in self.list_steps():
                raise ValueError(f"Etape inconnue (upto): {upto}")
            idx = self.list_steps().index(upto)
            comps = [c for c in comps if self.list_steps().index(c.name) <= idx]
        if only:
            if only not in self.list_steps():
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
        context["PIPELINE_PROFILE"] = "pipeline100ml"
        selected = self._select_components(only, upto, start)
        context["PIPELINE_STEPS"] = [c.name for c in selected]
        for comp in selected:
            output: Any = comp.run(context)
            if output is None:
                raise RuntimeError(f"{comp.name} a retourne None.")
        return context
