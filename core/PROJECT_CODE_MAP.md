# Project Code Map (DMS Core)

Date d'audit: 2026-03-05

## 1) Scope de l'audit
- Depot analyse: `/home/mourad/Bureau/DMS/core`
- Python files analyses: 19
- Fonctions/classes indexees: 346 (voir `FUNCTION_INDEX.txt`)
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
- Chaque entree `PRETRAITEMENT_RESULT[]` expose `size` (octets)

### 4.2 Routage OCR vs natif
- `TEXT_FILES`
- `IMAGE_ONLY_FILES`
- `DOCS`
- resume intermediaire: `PREPROCESS_RESULT`

### 4.3 Extraction texte
- sortie combinee OCR + natif: `FINAL_DOCS`
- documents natifs: `TEXT_DOCS`
- `FINAL_DOCS` / `TEXT_DOCS` propagent `size`

### 4.4 Tokenisation + layout
- sortie principale: `TOK_DOCS`
- alias utilise dans certains scripts: `selected`
- `TOK_DOCS` inclut `size` (source pour ES et fusion)

### 4.5 Attribution grammaticale (EN/FR/AR)
- consomme `selected`/`TOK_DOCS`/`FINAL_DOCS`
- ecrit des sorties structurees normalisees dans le context:
  - `NLP_ANALYSES`
  - `NLP_SENTENCES`
  - `NLP_ENTITIES`
  - `NLP_TOKENS` (sortie unifiee token-level: `filename`, `page_index`, `sent_index`, `tok_index`, `token`, `pos`, `lemma`, `ner`, `lang`)
    - normalisation punctuation: tokens `_` / `∅` (ou lemma `_` / `∅`) forces en `pos=PUNCT` avec lemma coherent
    - robustesse: nlp tokens conserve tous les tokens meme si `pos`/`lemmas` en entree sont plus courts
  - compat legacy: `NLP_POS` et `NLP_LEMMA` pointent vers la meme sortie unifiee
  - `NLP_LANGUAGE`, `NLP_LANGUAGE_STATS`, `DETECTED_LANGUAGES`

### 4.6 Elasticsearch (optionnel)
- activation: `USE_ELASTICSEARCH`
- conf: `ES_URL`, `ES_INDEX`
- auth HTTP: `ES_USER`, `ES_PASSWORD`, `ES_API_KEY`
- conf NLP: `ES_NLP_LEVEL` (`off|summary|full`), `ES_NLP_INDEX`, `ES_NLP_MAX_FULL_TOKENS`
- auto-start: `ES_AUTO_START`, `ES_START_COMMAND`, `ES_START_COMMANDS`, `ES_START_COMMAND_POSIX`, `ES_START_COMMAND_WINDOWS`, `ES_START_PASSWORD`, `ES_AUTO_START_WAIT_SECONDS`, `ES_AUTO_START_LAUNCH_TIMEOUT`
  - comportement par defaut: auto-start Windows uniquement, Linux/macOS en demarrage manuel.
- sorties: `ES_AVAILABLE`, `ES_DOC_IDS`, `ES_CLASSIFICATION_DOCS`, `ES_EXTRACTION_DOCS`, `ES_AUTO_STARTED`, `ES_AUTO_START_CMD`
- sorties NLP ES: `ES_NLP_SYNC`, `ES_NLP_DOCS_SYNCED`, `ES_NLP_TOKENS_SYNCED`, `ES_NLP_TOKEN_ERRORS`, `ES_NLP_LEVEL_EFFECTIVE`, `ES_NLP_INDEX_EFFECTIVE`

### 4.7 Classification
- sortie: `RESULTS` (doc_type, status, scores, `classification_log`, `keyword_matches`, `anti_confusion_targets`)
- sync ES: `ES_CLASSIFICATION_SYNCED`

### 4.8 Extraction regex
- sortie: `EXTRACTIONS`
- sync ES: `ES_EXTRACTION_SYNCED`

### 4.9 Fusion finale
- sortie fichier: `fusion_output.json`
- flags contexte: `FUSION_RESULT`, `FUSION_PAYLOAD`, `FUSION_PAYLOADS`, `FUSION_SOURCE`, `ES_FUSION_SYNCED`
- structure finale: `documents[]` (un bloc complet par document, avec `components`, `text`, `document_structure`, `extraction`, `nlp`, `quality_checks`)

## 5) Ou modifier selon le besoin

### 5.1 Changer la CLI, options, sequence des etapes
- `pipeline/cli.py`
- `pipeline/orchestrator.py`
- `pipeline/components.py`
- chargement `.env` global (optionnel): `pipeline/settings.py:load_dotenv` appele par `pipeline/cli.py`

### 5.2 Changer la detection text vs image
- `component/pretraitement-de-docs.py` (dont extraction `size`)
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
  - propagation `size` dans `TEXT_DOCS` et `FINAL_DOCS`

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
- `pipeline/elasticsearch.py`: indexation du champ documentaire `size`
- auto-demarrage local d'Elasticsearch quand indisponible: `pipeline/elasticsearch.py` (`_try_auto_start_elasticsearch`, `_resolve_auto_start_commands`, `maybe_build_store`)
- auto-start cross-platform: commandes dediees POSIX/Windows via variables d'environnement (ou `.env` optionnel)
- par defaut: Windows tente auto-start; Linux/macOS n'essaie pas sans commande explicite
- auth HTTP ES (Basic/API key): `ElasticsearchStore._request`
- sync NLP vers ES:
  - mode performant par defaut `summary` (stats linguistiques par document dans `dms_documents`)
  - mode `full` (tokens POS/lemma/NER dans index dedie, ex: `dms_nlp_tokens`)
  - fonction centrale: `sync_nlp_results` dans `pipeline/elasticsearch.py`

### 5.9 Changer fusion JSON finale
- `component/fusion_resultats.py`
- en mode `NLP full`, la fusion charge aussi les tokens NLP depuis ES (`dms_nlp_tokens`) pour le document courant et les structure par page/sentence.
- schema fusion lisible humain + exploitable code:
  - top-level: `schema_version`, `generated_at`, `source`, `documents_count`, `documents`
  - par document: `components` detaille tous les composants du pipeline
- classification detaillee exposee (`scores`, `classification_log`, `keyword_matches`, `anti_confusion_targets`) + `file.size`
- deduplication active: `components` reste volontairement compact (resume) et les details complets restent au niveau document (`classification`, `extraction`, `nlp`, `file`)

### 5.10 Changer logique linguistique EN/FR/AR
- orchestrateur langues: `component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py`
- anglais: `component/atrribution-gramatical/engcode.py`
- francais: `component/atrribution-gramatical/frcode.py`
- arabe: `component/atrribution-gramatical/arabcode.py`

## 6) Cartographie des fichiers Python (roles)

### 6.1 Orchestration (`pipeline/`)
- `pipeline/settings.py`: logging, normalize input, context managers cwd/argv, chargement `.env` optionnel.
- `pipeline/components.py`: wrappers des composants scripts + resumes + sync ES.
- `pipeline/orchestrator.py`: ordre des etapes, selection `only/upto/start`.
- `pipeline/cli.py`: CLI + chargement `.env` + tee print vers `outputgeneralterminal.txt`.
- `pipeline/elasticsearch.py`: store HTTP ES + auth + flatten/index + auto-start local ES (POSIX/Windows) + fallback docs + sync classification/extraction + sync NLP (summary/full).

### 6.2 Composants metier (`component/`)
- `pretraitement-de-docs.py`: detect format, determine `text` vs `image_only`, extrait `size`.
- `si-image-pretraiter-sinonpass-le-doc.py`: split OCR/native + preprocess image.
- `output-txt.py`: OCR tesseract + extraction native multi-format + `FINAL_DOCS` (+ `size`).
- `tokenisation-layout.py`: language detect + sentence/layout chunking + table/multicol + `TOK_DOCS` (+ `size`).
- `atrribution-gramatical/*.py`: POS/lemma/NER per language + notebook style runners.
- `elasticsearch.py`: step script pour index/fetch docs ES.
- `clasification.py`: keyword scoring classification + details matches (`classification_log`, `keyword_matches`).
- `extraction-regles.py`: regex extractors selon doc_type.
- `fusion_resultats.py`: build JSON fusion structure par document (`documents[]`) depuis context + ES.

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
  - `pipeline/components.py`: suppression de l'injection de `DOCS` placeholder pour les fichiers detectes `text` (evite la creation d'un faux document OCR vide).
  - `component/output-txt.py`: garde-fou sur la construction `FINAL_DOCS` pour ignorer les docs OCR vides et ne pas forcer `image_only` quand il n'y a aucune page OCR.
  - Effet valide sur `contrat_regex_test_corpus_fr_en_ar.pdf`: pretraitement `text=1 image=0`, preprocess `docs_prepped=0`, output-txt `1 docs | pages=12` avec `content='text'`.
  - `component/atrribution-gramatical/engcode.py`, `frcode.py`, `arabcode.py`: normalisation des sorties par phrase (`lang`, `tokens`, `pos`, `lemmas`, `ner_labels`, `entities`, metadata doc/page/sentence).
  - `component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py`:
    - correction de la reprise d'entree (ignore les listes vides),
    - publication de structures globales `NLP_ANALYSES`, `NLP_SENTENCES`, `NLP_ENTITIES`, `NLP_POS`, `NLP_LEMMA`, `NLP_LANGUAGE_STATS`.
  - `pipeline/cli.py`: nouvelles options `--es-nlp-level` et `--es-nlp-index`.
  - `pipeline/settings.py`: ajout `load_dotenv` pour charger automatiquement `.env` si present au lancement.
  - `pipeline/cli.py`:
    - lit maintenant les defaults depuis `.env` (`ES_URL`, `ES_INDEX`, `ES_NLP_*`, `USE_ELASTICSEARCH`),
    - ajoute les options `--es-user`, `--es-password`, `--es-api-key`,
    - passe `ES_AUTO_START*`/`ES_START_*`/`ES_START_PASSWORD` dans le context.
  - `pipeline/elasticsearch.py`:
    - ajout auth HTTP Basic/API key dans `ElasticsearchStore`,
    - auto-start compatible `.env` multi-commandes (`ES_START_COMMANDS=cmd1 || cmd2`) et commandes dediees POSIX/Windows,
    - support `sudo -S` via `ES_START_PASSWORD`,
    - parsing commandes plus robuste (`expandvars`, `shlex` adapte OS).
  - `pipeline/elasticsearch.py` + `pipeline/cli.py`:
    - comportement ajuste selon besoin utilisateur: auto-start par defaut uniquement sur Windows,
    - Linux/macOS en demarrage manuel Elasticsearch (pas de tentative auto sans commande explicite).
  - `pipeline/elasticsearch.py`:
    - correction compatibilite mapping ES existant: remplacement de `nlp.entities_sample` par `nlp.entities_sample_flat`,
    - evite l'erreur `mapper [nlp.entities_sample.text] cannot be changed from type [date] to [text]` sur index deja cree.
  - `pipeline/elasticsearch.py`:
    - ajout de `sync_nlp_results` (sync des analyses grammaticales vers ES),
    - mode `summary` (stats/doc) et mode `full` (index tokens dedie via bulk NDJSON),
    - nouvelles aides `_normalize_nlp_level`, `_extract_entities`, `_bulk_index_nlp_tokens`, etc.,
    - extension `ElasticsearchStore` (`ensure_custom_index`, `delete_by_query`, `bulk_ndjson`).
  - `component/elasticsearch.py`: appel de `sync_nlp_results` + exposition des compteurs `ES_NLP_*`.
  - `pipeline/components.py`: reporting enrichi du composant `elasticsearch` avec statut NLP.
  - `component/fusion_resultats.py`:
    - enrichissement de la sortie `fusion_output.json` avec les donnees NLP ES par document,
    - section `nlp.summary` alimentee depuis `dms_documents.nlp`,
    - section `nlp.full` (mode full) contenant:
      - `index`, `doc_id`, `count`, `returned`, `truncated`,
      - `tokens` (liste a plat unifiee `filename/page_index/sent_index/tok_index/token/pos/lemma/ner/lang`),
      - `structure` (groupement par `page_index` puis `sent_index`).
    - fallback intelligent par `filename` quand l'identifiant document diverge entre index docs et index tokens.
  - `component/fusion_resultats.py`:
    - refonte de la structure finale `fusion_output.json` pour une lecture humaine + usage code:
      - toujours un tableau `documents[]`,
      - un bloc `components` par document (`pretraitement`, `ocr routing`, `output_txt`, `tokenisation`, `attribution_grammaticale`, `elasticsearch`, `classification`, `extraction_regles`, `fusion`),
      - conservation des donnees metier (`text`, `document_structure`, `extraction`, `nlp`, `quality_checks`) sous le meme document.
  - `component/fusion_resultats.py`:
    - suppression de redondances dans la sortie finale:
      - suppression de l'alias top-level `document` (doublon de `documents[0]`),
      - suppression de doublons volumineux dans `components` (`doc`, `prepared_doc`, payloads complets classification/extraction),
      - conservation de resumes compacts dans `components` + donnees detaillees uniques dans `classification` / `extraction` / `nlp`.
  - `component/pretraitement-de-docs.py`: `PRETRAITEMENT_RESULT[]` inclut maintenant `size` (octets).
  - `component/output-txt.py`: propagation de `size` dans `TEXT_DOCS` et `FINAL_DOCS`.
  - `component/tokenisation-layout.py`: propagation de `size` dans `DOC_PACK` puis `TOK_DOCS`.
  - `pipeline/elasticsearch.py`: ajout du champ `size` dans les documents indexes (`flatten_tok_doc_for_index` + mapping `size: long`).
  - `component/clasification.py`:
    - `RESULTS` enrichi avec `winning_score`,
    - ajout `classification_log`, `keyword_matches` (strong/medium/weak/negative/strong_negative/anti_confusion_hits),
    - ajout `anti_confusion_targets`.
  - `component/fusion_resultats.py`:
    - remplit `file.size` depuis contexte/pretraitement/ES,
    - enrichit `classification` au niveau document (scores/log/matches/anti-confusion),
    - corrige la vue `components.pretraitement_de_docs` (ext/mime/label/content),
    - en mode ES: priorite aux donnees `classification`/`rule_extraction` deja stockees dans ES.
  - `component/fusion_resultats.py`:
    - optimisation anti-redondance supplementaire: retrait des champs lourds dupliques dans `components` (`scores`, `keyword_matches`, `classification_log`, `anti_confusion_targets`, tailles dupliquees),
    - conservation des details complets uniquement dans les blocs principaux du document.
  - `graphecode.html`:
    - refonte complete en graphe runtime pipeline (commande cible `main.py ... --use-elasticsearch --es-nlp-level full`),
    - ajout de 2 scenarios visuels (`es_off` reel fallback local / `es_on` cible ES disponible),
    - ajout des vues explicatives `Sequence`, `Context Keys`, `Run Trace`,
    - clic sur un composant: affichage d'un output exemple + explication de chaque champ.
- 2026-03-15:
  - `component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py`:
    - fusion des anciennes sorties separees `NLP_POS` et `NLP_LEMMA` en une sortie unique `NLP_TOKENS`,
    - format unifie par token: `filename`, `page_index`, `sent_index`, `tok_index`, `token`, `pos`, `lemma`, `ner`, `lang`,
    - normalisation punctuation: `_` / `∅` (token ou lemma) force `pos=PUNCT`,
    - ne drop plus les tokens quand longueurs `tokens`, `pos`, `lemmas` divergent,
    - maintien compatibilite descendante: `NLP_POS` et `NLP_LEMMA` aliases de `NLP_TOKENS`.
  - `component/fusion_resultats.py`:
    - consommation prioritaire de `NLP_TOKENS` pour la sortie `nlp.tokens`,
    - fallback automatique pour anciens runs (merge `NLP_POS` + `NLP_LEMMA` par cle token),
    - suppression de la redondance de sortie en separant plus `nlp.pos`/`nlp.lemma` (une seule liste `nlp.tokens`).

## 12) Regle de maintenance
- A chaque modification de code Python dans `pipeline/` ou `component/`:
  - mettre a jour `FUNCTION_INDEX.txt`
  - ajouter/mettre a jour l'entree correspondante dans `PROJECT_CODE_MAP.md` (sections impactees + changelog)

## 13) Visualisation HTML du pipeline
- Fichier: `graphecode.html`
- Role: graphe runtime interactif du pipeline `core` pour la commande:
  - `python main.py documents/contrat_regex_test_corpus_fr_en_ar.pdf --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens`
- Contenu de la vue:
  - noeuds execution (CLI -> orchestrateur -> 9 composants)
  - noeuds scripts (`component/*.py` et wrappers `pipeline/components.py`)
  - noeuds de contexte (`INPUT_FILE`, `FINAL_DOCS`, `TOK_DOCS`, `NLP_*`, `RESULTS`, `EXTRACTIONS`, `FUSION_*`, `ES_*`)
  - branches conditionnelles:
    - `content == image_only ?`
    - `ES ping OK ?`
    - `es-nlp-level == full ?`
  - sorties/stockage:
    - `outputgeneralterminal.txt`
    - `fusion_output.json`
    - index ES `dms_documents`
    - index ES `dms_nlp_tokens`
- Scenarios integres dans la meme page:
  - `es_off`: run reel capture (fallback local)
  - `es_on`: run cible ES disponible + NLP full
- Tables explicatives integrees:
  - sequence pipeline
  - map des context keys
  - trace run horodatee (2026-03-05)
- Interaction detaillee au clic composant:
  - affiche un `output` exemple realiste du composant
  - affiche une explication courte champ-par-champ
