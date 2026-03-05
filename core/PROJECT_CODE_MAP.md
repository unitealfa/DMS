# Project Code Map (DMS Core)

Date d'audit: 2026-03-05

## 1) Scope de l'audit
- Depot analyse: `/home/mourad/Bureau/DMS/core`
- Python files analyses: 19
- Fonctions/classes indexees: 305 (voir `FUNCTION_INDEX.txt`)
- Regles metier JSON: `rules/*.json` + `classification/*.json` + `config/ruleset_routes.json`

## 2) Points d'entree
- CLI principal: `main.py` -> `pipeline.cli:main`
- CLI package: `orchestre` (defini dans `pyproject.toml`)
- Parsing des options: `pipeline/cli.py`
- Orchestrateur: `pipeline/orchestrator.py`
- Wrappers d'execution des composants: `pipeline/components.py`

## 3) Pipeline reel (ordre d'execution)
1. `pretraitement-de-docs`
2. `si-image-pretraiter-sinonpass-le-doc`
3. `output-txt`
4. `tokenisation-layout`
5. `atripusion-gramatical-en-utilisant-les3ficherla`
6. `elasticsearch`
7. `clasification`
8. `extraction-regles`
9. `fusion-resultats` (debug/fusion finale, non bloquant en erreur)

Reference implementation de l'ordre: `pipeline/orchestrator.py`

## 4) Flux des donnees (context globals)
- Le pipeline repose sur un `context` dict partage entre composants (exec via `runpy.run_path`).
- Cles majeures produites/consommees:

### 4.1 Entree / pretraitement
- Entree user: `INPUT_FILE`
- Sortie pretraitement type de contenu: `PRETRAITEMENT_RESULT`

### 4.2 Routage OCR vs natif
- `TEXT_FILES`
- `IMAGE_ONLY_FILES`
- `DOCS`
- resume intermediaire: `PREPROCESS_RESULT`

### 4.3 Extraction texte
- sortie combinee OCR + natif: `FINAL_DOCS`
- documents natifs: `TEXT_DOCS`

### 4.4 Tokenisation + layout
- sortie principale: `TOK_DOCS`
- alias utilise dans certains scripts: `selected`

### 4.5 Attribution grammaticale (EN/FR/AR)
- consomme `selected`/`TOK_DOCS`/`FINAL_DOCS`
- affiche des resultats linguistiques (pas de structure unique normalisee ecrite dans context)

### 4.6 Elasticsearch (optionnel)
- activation: `USE_ELASTICSEARCH`
- conf: `ES_URL`, `ES_INDEX`
- auto-start: `ES_AUTO_START`, `ES_START_COMMAND`, `ES_START_COMMANDS`, `ES_AUTO_START_WAIT_SECONDS`, `ES_AUTO_START_LAUNCH_TIMEOUT`
- sorties: `ES_AVAILABLE`, `ES_DOC_IDS`, `ES_CLASSIFICATION_DOCS`, `ES_EXTRACTION_DOCS`, `ES_AUTO_STARTED`, `ES_AUTO_START_CMD`

### 4.7 Classification
- sortie: `RESULTS` (doc_type, status, scores)
- sync ES: `ES_CLASSIFICATION_SYNCED`

### 4.8 Extraction regex
- sortie: `EXTRACTIONS`
- sync ES: `ES_EXTRACTION_SYNCED`

### 4.9 Fusion finale
- sortie fichier: `fusion_output.json`
- flags contexte: `FUSION_RESULT`, `FUSION_PAYLOAD`, `FUSION_PAYLOADS`, `FUSION_SOURCE`, `ES_FUSION_SYNCED`

## 5) Ou modifier selon le besoin

### 5.1 Changer la CLI, options, sequence des etapes
- `pipeline/cli.py`
- `pipeline/orchestrator.py`
- `pipeline/components.py`

### 5.2 Changer la detection text vs image
- `component/pretraitement-de-docs.py`
- `component/si-image-pretraiter-sinonpass-le-doc.py`

### 5.3 Changer preprocessing OCR (contrast, threshold, rotate, etc.)
- `component/si-image-pretraiter-sinonpass-le-doc.py`
  - `EnhanceOptions`
  - `preprocess_image`
  - `parse_args`

### 5.4 Changer extraction texte natif (PDF/DOCX/XLSX/PPTX/ODF/EPUB)
- `component/output-txt.py`
  - `extract_text_native`
  - helpers `_docx_xml_to_text`, `_xlsx_sheet_to_text`, etc.

### 5.5 Changer segmentation layout / tables / multi-colonnes / bruit
- `component/tokenisation-layout.py`
  - `layout_items`
  - `_transpose_or_group_multicol`
  - `_collect_table_block`
  - `chunk_layout_universal`
  - `chunk_is_noise`

### 5.6 Changer classification documentaire (scores, threshold, priorites)
- code: `component/clasification.py`
- config: `classification/common.json`, `classification/*.json`

### 5.7 Changer extraction regex metier
- moteur: `component/extraction-regles.py`
- routage rulesets: `config/ruleset_routes.json`
- patterns metier: `rules/*.json`

### 5.8 Changer logique Elasticsearch
- composant pont: `component/elasticsearch.py`
- client + mapping + index/update logic: `pipeline/elasticsearch.py`
- auto-demarrage local d'Elasticsearch quand indisponible: `pipeline/elasticsearch.py` (`_try_auto_start_elasticsearch`, `_resolve_auto_start_commands`, `maybe_build_store`)

### 5.9 Changer fusion JSON finale
- `component/fusion_resultats.py`

### 5.10 Changer logique linguistique EN/FR/AR
- orchestrateur langues: `component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py`
- anglais: `component/atrribution-gramatical/engcode.py`
- francais: `component/atrribution-gramatical/frcode.py`
- arabe: `component/atrribution-gramatical/arabcode.py`

## 6) Cartographie des fichiers Python (roles)

### 6.1 Orchestration (`pipeline/`)
- `pipeline/settings.py`: logging, normalize input, context managers cwd/argv.
- `pipeline/components.py`: wrappers des composants scripts + resumes + sync ES.
- `pipeline/orchestrator.py`: ordre des etapes, selection `only/upto/start`.
- `pipeline/cli.py`: CLI + tee print vers `outputgeneralterminal.txt`.
- `pipeline/elasticsearch.py`: store HTTP ES + flatten/index + auto-start local ES + fallback docs + sync classification/extraction.

### 6.2 Composants metier (`component/`)
- `pretraitement-de-docs.py`: detect format, determine `text` vs `image_only`.
- `si-image-pretraiter-sinonpass-le-doc.py`: split OCR/native + preprocess image.
- `output-txt.py`: OCR tesseract + extraction native multi-format + `FINAL_DOCS`.
- `tokenisation-layout.py`: language detect + sentence/layout chunking + table/multicol + TOK_DOCS.
- `atrribution-gramatical/*.py`: POS/lemma/NER per language + notebook style runners.
- `elasticsearch.py`: step script pour index/fetch docs ES.
- `clasification.py`: keyword scoring classification.
- `extraction-regles.py`: regex extractors selon doc_type.
- `fusion_resultats.py`: build JSON fusion from context or ES.

## 7) Fichiers metier JSON
- `classification/common.json`: poids/penalites/seuil/marge globaux.
- `classification/*.json`: classes documentaires + keywords + anti-confusion.
- `config/ruleset_routes.json`: mapping doc_type -> rulesets.
- `rules/common.json`: extracteurs communs (date/email/phone/url).
- `rules/FACTURE.json`: extracteurs facture FR/EN/AR.
- `rules/BON_DE_COMMANDE.json`: extracteurs BC FR/EN/AR.
- `rules/CONTRAT.json`: extracteurs contrat FR/EN/AR.

## 8) Artefacts d'execution
- `orchestre.log`: log Python (logging)
- `outputgeneralterminal.txt`: tee des `print(...)`
- `fusion_output.json`: sortie fusion finale

## 9) Notes de qualite observees pendant audit
- `component/si-image-pretraiter-sinonpass-le-doc.py` contient du code duplique sur la construction de `DOCS` (deux boucles consecutives).
- `component/extraction-regles.py` compile les regex sans garde plus fine que `re.error`; en cas de pattern invalide, le champ est skippe (comportement tolerant).
- `component/fusion_resultats.py` est clairement oriente "debug/fusion", pas schema strict valide via validation formelle.
- La couche `pipeline/` est proprement separee et sert de facade stable autour des scripts notebooks.

## 10) Index complet des fonctions/classes
- Voir `FUNCTION_INDEX.txt` pour la liste exhaustive `file:line:def/class`.
- Ce fichier est la reference la plus rapide pour localiser une modification precise.

## 11) Changelog code
- 2026-03-05:
  - `pipeline/elasticsearch.py`: ajout d'un auto-demarrage Elasticsearch local avant fallback.
  - Nouvelles fonctions: `_safe_positive_int`, `_is_local_es_url`, `_normalize_command`, `_format_command`, `_resolve_auto_start_commands`, `_run_auto_start_command`, `_wait_for_es_ping`, `_try_auto_start_elasticsearch`.
  - `maybe_build_store` tente maintenant un demarrage auto puis re-ping avant `Fallback sur flux local`.
  - `component/clasification.py`: correction de `_get_previous_cell_input` pour ignorer les listes vides (ex: `ES_CLASSIFICATION_DOCS=[]` quand ES indisponible) et continuer vers `selected`/`TOK_DOCS`.
  - `component/extraction-regles.py`: correction de `_get_input_docs` pour ignorer les listes vides (ex: `ES_EXTRACTION_DOCS=[]`) et prendre la source suivante.
  - Effet: suppression du crash `Format d'entree non supporte: <class 'list'>` en mode `--use-elasticsearch` avec ES down.
  - `pipeline/elasticsearch.py`: ajout de `_same_es_target` et memo de disponibilite dans `maybe_build_store` pour eviter les warnings repetes sur les composants suivants (notamment `fusion-resultats`) quand ES est deja marque indisponible.

## 12) Regle de maintenance
- A chaque modification de code Python dans `pipeline/` ou `component/`:
  - mettre a jour `FUNCTION_INDEX.txt`
  - ajouter/mettre a jour l'entree correspondante dans `PROJECT_CODE_MAP.md` (sections impactees + changelog)
