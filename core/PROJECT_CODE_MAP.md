# Project Code Map (DMS Core)

Date d'audit: 2026-04-08

## 1) Scope de l'audit
- Depot analyse: `/home/mourad/Bureau/DMS/core`
- Python files analyses: `40`
- Fonctions/classes indexees: `883` (voir [FUNCTION_INDEX.txt](/home/mourad/Bureau/DMS/core/FUNCTION_INDEX.txt))
- Regles metier:
  - `rules/*.json`
  - `rules/*.yaml`
  - `classification/*.json`
  - `config/ruleset_routes.json`
  - `config/ruleset_routes.yaml`

## 2) Points d'entree
- CLI principal:
  - [main.py](/home/mourad/Bureau/DMS/core/main.py)
  - pointe vers `pipeline.cli:main`
- API locale:
  - [local_api.py](/home/mourad/Bureau/DMS/core/local_api.py)
  - pointe vers `pipeline.local_api:main`
- CLI package:
  - `orchestre`
  - defini dans [pyproject.toml](/home/mourad/Bureau/DMS/core/pyproject.toml)
- API locale package:
  - `dms-local-api`
  - defini dans [pyproject.toml](/home/mourad/Bureau/DMS/core/pyproject.toml)

## 3) Couches runtime principales
- [pipeline/cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py)
  - parsing CLI
  - selection de pipeline
  - options Elasticsearch
  - `PIPELINE_DEFAULT_CODE`
- [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
  - `BasePipelineOrchestrator`
  - pipelines concretes
  - registre dynamique des pipelines
  - resolution `default` / aliases / codes
- [pipeline/components.py](/home/mourad/Bureau/DMS/core/pipeline/components.py)
  - wrappers des composants
  - execution via `runpy.run_path`
  - resumes terminal
  - wrapper generique `Component` executable
- [pipeline/component_trace.py](/home/mourad/Bureau/DMS/core/pipeline/component_trace.py)
  - trace runtime des composants
  - cles touchees
  - resume/output reporte
- [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py)
  - serveur HTTP local
  - upload
  - lancement pipeline
  - polling status
  - retour du resultat final API
- [pipeline/runtime_state.py](/home/mourad/Bureau/DMS/core/pipeline/runtime_state.py)
  - etat runtime job/pipeline/composant

## 4) Pipelines enregistrees
Le systeme n'est plus limite a une liste statique codee en dur dans la CLI.

Pipelines actuellement presentes dans le depot:
- `pipeline0ml`
- `pipeline50ml`
- `pipeline100ml`

Pipeline par defaut actuelle:
- `PIPELINE_DEFAULT_CODE = "pipeline50ml"`
- defini dans [pipeline/cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py)

Registre dynamique:
- `pipeline_orchestrator_classes()`
- `pipeline_registry()`
- `available_pipeline_codes()`
- `available_pipeline_choices()`
- `normalize_pipeline_name()`
- `create_pipeline_orchestrator()`
- tout cela est dans [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)

## 5) Ordre reel des pipelines

### 5.1 `pipeline0ml`
1. `pretraitement-de-docs`
2. `si-image-pretraiter-sinonpass-le-doc`
3. `output-txt`
4. `clasification`
5. `tokenisation-layout`
6. `atripusion-gramatical`
7. `table-extraction`
8. `verification-totaux`
9. `liaison-inter-docs`
10. `elasticsearch`
11. `extraction-regles`
12. `fusion-resultats`
13. `api-output`

### 5.2 `pipeline50ml`
1. `pretraitement-de-docs`
2. `si-image-pretraiter-sinonpass-le-doc`
3. `output-txt`
4. `clasification`
5. `tokenisation-layout` (`tokenisation-layout-50ml.py`)
6. `atripusion-gramatical`
7. `table-extraction`
8. `verification-totaux`
9. `liaison-inter-docs`
10. `elasticsearch`
11. `extraction-regles` (`extraction-regles-50ml.py`)
12. `fusion-resultats`
13. `api-output`

### 5.3 `pipeline100ml`
1. `pretraitement-de-docs`
2. `si-image-pretraiter-sinonpass-le-doc`
3. `output-txt`
4. `clasification`
5. `tokenisation-layout` (`tokenisation-layout-100ml.py`)
6. `atripusion-gramatical` (`attribution-gramatical-100ml-xlmr.py`)
7. `table-extraction`
8. `verification-totaux`
9. `detection-signature-chachet-codebarr`
10. `liaison-inter-docs`
11. `elasticsearch`
12. `extraction-regles` (`extraction-regles-100ml.py`)
13. `fusion-resultats`
14. `api-output`

## 6) Flux des donnees
Le pipeline repose sur un dictionnaire partage `context`.

Entree:
- `INPUT_FILE`

Sorties intermediaires majeures:
- `PRETRAITEMENT_RESULT`
- `TEXT_FILES`
- `IMAGE_ONLY_FILES`
- `DOCS`
- `PREPROCESS_RESULT`
- `FINAL_DOCS`
- `RESULTS`
- `TOK_DOCS`
- `selected`
- `NLP_ANALYSES`
- `NLP_SENTENCES`
- `NLP_TOKENS`
- `NLP_ENTITIES`
- `TABLE_EXTRACTIONS`
- `TOTALS_VERIFICATION`
- `VISUAL_MARKS_DETECTIONS`
- `INTERDOC_ANALYSIS`
- `EXTRACTIONS`
- `FUSION_RESULT`
- `FUSION_PAYLOAD`
- `FUSION_PAYLOADS`
- `API_OUTPUT_RESULT`

Passage entre composants:
- execution via `runpy.run_path(..., init_globals=context)` dans [pipeline/components.py](/home/mourad/Bureau/DMS/core/pipeline/components.py)
- le composant suivant reutilise directement les cles ecrites par le precedent

## 7) Role des grands composants

### 7.1 Pretraitement
- [component/pretraitement-de-docs.py](/home/mourad/Bureau/DMS/core/component/pretraitement-de-docs.py)
- detecte le format
- classe grossierement `text` / `image_only` / `unsupported`

### 7.2 Routage OCR / natif
- [component/si-image-pretraiter-sinonpass-le-doc.py](/home/mourad/Bureau/DMS/core/component/si-image-pretraiter-sinonpass-le-doc.py)
- separe texte natif et image OCR
- prepare `TEXT_FILES`, `IMAGE_ONLY_FILES`, `DOCS`

### 7.3 Extraction texte
- [component/output-txt.py](/home/mourad/Bureau/DMS/core/component/output-txt.py)
- extrait le texte final exploitable
- gere PDF, DOCX, HTML, images OCR, XLSX

### 7.4 Classification
- [component/clasification.py](/home/mourad/Bureau/DMS/core/component/clasification.py)
- classement documentaire par regles/mots-cles
- sortie `RESULTS`

### 7.5 Tokenisation / layout
- [component/tokenisation_layout/tokenisation-layout.py](/home/mourad/Bureau/DMS/core/component/tokenisation_layout/tokenisation-layout.py)
- [component/tokenisation_layout/tokenisation-layout-50ml.py](/home/mourad/Bureau/DMS/core/component/tokenisation_layout/tokenisation-layout-50ml.py)
- [component/tokenisation_layout/tokenisation-layout-100ml.py](/home/mourad/Bureau/DMS/core/component/tokenisation_layout/tokenisation-layout-100ml.py)
- version 50ml:
  - vecteurs/hash/topics legers
- version 100ml:
  - embeddings Transformer
  - topics document/chunk

### 7.6 Grammaire / NLP
- [component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py](/home/mourad/Bureau/DMS/core/component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py)
- [component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py](/home/mourad/Bureau/DMS/core/component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py)
- produit POS, lemma, NER, langue
- sorties:
  - `NLP_ANALYSES`
  - `NLP_SENTENCES`
  - `NLP_TOKENS`
  - `NLP_ENTITIES`

### 7.7 Tableaux
- [component/table_extraction/table-extraction.py](/home/mourad/Bureau/DMS/core/component/table_extraction/table-extraction.py)
- [component/table_extraction/table_extraction_lib.py](/home/mourad/Bureau/DMS/core/component/table_extraction/table_extraction_lib.py)
- detecte tableaux et lignes utiles produit/quantite/prix/total
- sorties:
  - `TABLE_EXTRACTIONS`
  - `TABLE_EXTRACTIONS_50ML`
  - `TABLE_EXTRACTIONS_100ML`

### 7.8 Verification des totaux
- [component/verification-totaux.py](/home/mourad/Bureau/DMS/core/component/verification-totaux.py)
- compare lignes et totaux declares
- sortie `TOTALS_VERIFICATION`

### 7.9 Detection visuelle
- [component/detection-signature-chachet-codebarr.py](/home/mourad/Bureau/DMS/core/component/detection-signature-chachet-codebarr.py)
- uniquement dans `pipeline100ml`
- detecte:
  - signatures
  - cachets
  - QR
  - codes-barres

### 7.10 Liaison inter-documents
- [component/liaison-inter-docs.py](/home/mourad/Bureau/DMS/core/component/liaison-inter-docs.py)
- rapprochement documents par topics, phrases, vecteurs si disponibles
- sortie `INTERDOC_ANALYSIS`

### 7.11 Elasticsearch
- [component/elasticsearch.py](/home/mourad/Bureau/DMS/core/component/elasticsearch.py)
- indexation / sync optionnelle
- index principal par defaut: `dms_documents`
- index NLP full par defaut: `dms_nlp_tokens`

### 7.12 Extraction metier
- [component/extraction/extraction-regles.py](/home/mourad/Bureau/DMS/core/component/extraction/extraction-regles.py)
- [component/extraction/extraction-regles-50ml.py](/home/mourad/Bureau/DMS/core/component/extraction/extraction-regles-50ml.py)
- [component/extraction/extraction-regles-100ml.py](/home/mourad/Bureau/DMS/core/component/extraction/extraction-regles-100ml.py)
- version standard:
  - extraction regles/regex
- versions 50ml/100ml:
  - YAML + BM25 maison

### 7.13 Fusion finale
- [component/fusion_resultats.py](/home/mourad/Bureau/DMS/core/component/fusion_resultats.py)
- construit `fusion_output.json`
- expose `FUSION_RESULT`, `FUSION_PAYLOAD`, `FUSION_PAYLOADS`
- auto-expose aussi les sorties des nouveaux composants traces

### 7.14 Sortie API finale
- [component/api-output.py](/home/mourad/Bureau/DMS/core/component/api-output.py)
- normalise la sortie finale selon:
  - [dms-unified-output-template.json](/home/mourad/Bureau/DMS/core/dms-unified-output-template.json)
- ecrit:
  - `api_storage/uploads/<job_id>/result.json`
- peut aussi POSTer le resultat complet vers `callback_url`

## 8) Extensibilite

### 8.1 Ajouter un composant
- creer le script dans `component/`
- l'ajouter dans une pipeline dans [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
- pour un composant standard, le wrapper generique `Component(...)` suffit
- la CLI, `fusion-resultats` et `api-output` recuperent alors automatiquement sa trace runtime

### 8.2 Ajouter une pipeline
- creer une nouvelle classe heritant de `BasePipelineOrchestrator`
- definir:
  - `code`
  - `aliases`
  - `label`
  - `description`
  - `build_components()`
- ensuite:
  - la CLI la voit automatiquement
  - `local_api.py` la voit automatiquement
  - `default` peut pointer dessus via `PIPELINE_DEFAULT_CODE`

## 9) API locale
Serveur:
- [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py)

Fichiers stockes:
- `api_storage/uploads/<job_id>/`

Endpoints principaux:
- `GET /`
- `POST /api/run`
- `POST /api/store`
- `GET /api/status`
- `GET /api/result/<job_id>`
- `GET /api/documents`
- `GET /api/documents/<job_id>`
- `GET /api/documents/file/<job_id>/<filename>`

Resultat final API:
- `result.json`
- schema base sur [dms-unified-output-template.json](/home/mourad/Bureau/DMS/core/dms-unified-output-template.json)
- champs absents laisses a `null` ou `[]` selon le template

## 10) Sorties importantes sur disque
- log terminal principal:
  - `outputgeneralterminal.txt`
- log terminal runtime si le document d'entree est `outputgeneralterminal.txt`:
  - `outputgeneralterminal.runtime.txt`
- fusion debug:
  - `fusion_output.json`
- resultat API:
  - `api_storage/uploads/<job_id>/result.json`
- manifest API:
  - `api_storage/uploads/<job_id>/manifest.json`

## 11) Technologies reellement utilisees
- OCR:
  - `Tesseract`
- pipeline 100ml:
  - `transformers`
  - `xlm-roberta-base`
- pipeline 50ml:
  - mode `fasttext-like` local par hashing subword
- BM25:
  - seulement dans `extraction-regles-50ml.py`
  - et `extraction-regles-100ml.py`
- Word2Vec:
  - non utilise
- PostgreSQL:
  - retire du code actuel

## 12) Fichiers de documentation interne
- [README.md](/home/mourad/Bureau/DMS/core/README.md)
  - doc utilisateur + API + extensibilite
- [EXPLICATION_PIPELINES.txt](/home/mourad/Bureau/DMS/core/EXPLICATION_PIPELINES.txt)
  - vue rapide des pipelines
- [FUNCTION_INDEX.txt](/home/mourad/Bureau/DMS/core/FUNCTION_INDEX.txt)
  - index exhaustif des fonctions/classes
- [PROJECT_CODE_MAP.md](/home/mourad/Bureau/DMS/core/PROJECT_CODE_MAP.md)
  - cartographie technique resumee

## 13) Regle de maintenance
A chaque changement dans `pipeline/` ou `component/`:
- mettre a jour [FUNCTION_INDEX.txt](/home/mourad/Bureau/DMS/core/FUNCTION_INDEX.txt)
- mettre a jour [PROJECT_CODE_MAP.md](/home/mourad/Bureau/DMS/core/PROJECT_CODE_MAP.md)
- mettre a jour [README.md](/home/mourad/Bureau/DMS/core/README.md) si le comportement externe change
