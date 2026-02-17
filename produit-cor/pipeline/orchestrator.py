from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from .components import (
    ClassificationComponent,
    GrammarComponent,
    OCRPreprocessComponent,
    OutputTxtComponent,
    PretraitementComponent,
    TokenisationLayoutComponent,
)
from .settings import COMPONENT_DIR, Context, InputLike, normalize_input


class PipelineOrchestrator:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.components = [
            PretraitementComponent("pretraitement-de-docs", COMPONENT_DIR / "pretraitement-de-docs.py"),
            OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", COMPONENT_DIR / "si-image-pretraiter-sinonpass-le-doc.py"),
            OutputTxtComponent("output-txt", COMPONENT_DIR / "output-txt.py"),
            TokenisationLayoutComponent("tokenisation-layout", COMPONENT_DIR / "tokenisation-layout.py"),
            GrammarComponent("atripusion-gramatical-en-utilisant-les3ficherla", COMPONENT_DIR / "atrribution-gramatical" / "atripusion-gramatical-en-utilisant-les3ficherla.py"),
            ClassificationComponent("clasification", COMPONENT_DIR / "clasification.py"),
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

    def run(self, input_files: InputLike, only: Optional[str] = None, upto: Optional[str] = None, start: Optional[str] = None) -> Context:
        context: Context = {"INPUT_FILE": normalize_input(input_files)}
        selected = self._select_components(only, upto, start)
        for comp in selected:
            output: Any = comp.run(context)
            if output is None:
                raise RuntimeError(f"{comp.name} a retourne None.")
        return context
