# DMS Pipeline Orchestrator

Ce dépôt regroupe des scripts de traitement documentaire (prétraitement, OCR, tokenisation, grammaire, classification) et un orchestrateur léger qui les enchaîne **sans modifier leur logique métier**.

## Architecture
- `pretraitement-de-docs.py` → `si-image-pretraiter-sinonpass-le-doc` → `output-txt.py` → `tokenisation-layout` → `atripusion-gramatical-en-utilisant-les3ficherla.py` → `clasification.py`
- `pipeline/` : couche d'orchestration open-source friendly  
  - `settings.py` : logging, helpers (argv isolation, cwd, normalisation des entrées)  
  - `components.py` : wrappers `Component` pour chaque script  
  - `orchestrator.py` : assemble l'ordre des composants  
  - `cli.py` : parsing CLI et point d'entrée
- `orchestre.py` : shim pour lancer le CLI (`python orchestre.py ...` ou `orchestre ...` via console_script).

## Exécution
```bash
python orchestre.py documents/englais.docx
# ou
python -m pipeline.cli documents/englais.docx
```

## Maintenance / Open Source
- Code orchestrateur typé et découpé par responsabilités (helpers vs composants vs CLI).
- Pas de dépendance aux chemins Windows dans le code d'orchestration (chemins relatifs au repo).
- Journalisation dans `orchestre.log`.
- Paquet installable : `pip install -e .` puis `orchestre ...`.

## Règles respectées
- Aucun algorithme interne des scripts métiers n'est modifié ni copié.
- Orchestration uniquement : passage des sorties en entrées, validations, logs lisibles.
