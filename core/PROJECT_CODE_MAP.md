# Project Code Map (DMS Core)

Date d'audit: 2026-03-19

## 1) Scope de l'audit
- Depot analyse: `/home/mourad/Bureau/DMS/core`
- Python files analyses: 32
- Fonctions/classes indexees: 757 (voir `FUNCTION_INDEX.txt`)
- Regles metier JSON/YAML: `rules/*.json` + `rules/*.yaml` + `classification/*.json` + `config/ruleset_routes.json` + `config/ruleset_routes.yaml`
- Note historique: les entrees de changelog anterieures au `2026-03-19` peuvent citer les anciens chemins plats sous `component/` avant le refactoring en sous-dossiers.

## 2) Points d'entree
- CLI principal: `main.py` -> `pipeline.cli:main`
- CLI package: `orchestre` (defini dans `pyproject.toml`)
- Parsing des options: `pipeline/cli.py`
- Orchestrateurs:
  - `pipeline/orchestrator.py` (contient `PipelineOrchestrator` + `Pipeline50MLOrchestrator` + `Pipeline100MLOrchestrator`)
- Wrappers d'execution des composants: `pipeline/components.py`
- Document de lecture rapide des 3 pipelines: `EXPLICATION_PIPELINES.txt`

## 3) Pipeline reel (ordre d'execution)
- Pipeline `default`:
  1. `pretraitement-de-docs`
  2. `si-image-pretraiter-sinonpass-le-doc`
  3. `output-txt`
  4. `clasification`
  5. `tokenisation-layout`
  6. `atripusion-gramatical` (`component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py`):
     - attribution grammaticale legacy basee sur les 3 modules EN/FR/AR (`engcode.py`, `frcode.py`, `arabcode.py`).
  7. `table-extraction` (`component/table_extraction/table-extraction.py`):
     - extraction de tableaux non-ML pour la pipeline standard,
     - detection par heuristiques de geometrie + synonymes/metiers de colonnes (`product`, `quantity`, `unit_price`, `total`, etc.),
     - sortie context: `TABLE_EXTRACTIONS_DEFAULT` + `TABLE_EXTRACTIONS`.
  8. `verification-totaux` (`component/verification-totaux.py`):
     - verification non-ML des sous-totaux, taxes et totaux a partir des lignes/tableaux extraits,
     - audit ligne par ligne (`quantity * unit_price`) + comparaison aux totaux declares les plus plausibles,
     - sortie context: `TOTALS_VERIFICATION`.
  9. `liaison-inter-docs`
  10. `elasticsearch`
  11. `extraction-regles`
  12. `fusion-resultats` (debug/fusion finale, non bloquant en erreur)
- Pipeline `pipeline50ml`:
  1. `pretraitement-de-docs`
  2. `si-image-pretraiter-sinonpass-le-doc`
  3. `output-txt`
  4. `clasification`
  5. `tokenisation-layout` (`component/tokenisation_layout/tokenisation-layout-50ml.py`):
     - tokenisation/layout standard + enrichissement FastText-like (subword hashing),
     - vecteurs mot/chunk/document,
     - topic extraction par chunk (`chunk_primary_topic`, `chunk_topics`) + topics document,
     - generation `NLP_*` minimale (provisoire, puis remplacee par la sortie grammaire).
  6. `atripusion-gramatical` (`component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py`)
  7. `table-extraction` (`component/table_extraction/table-extraction.py`):
     - extraction tableaux unifiee pour 50ml/100ml (moteur commun),
     - detection agnostique renforcee par geometrie texte (ancrages X stables + score de tabularite par ligne),
     - meilleure detection des tableaux denses/serres OCR,
     - sortie context: `TABLE_EXTRACTIONS_50ML` + `TABLE_EXTRACTIONS`.
  8. `verification-totaux` (`component/verification-totaux.py`):
     - verification non-ML partagee avec les autres pipelines,
     - consomme `TABLE_EXTRACTIONS_50ML` / `TABLE_EXTRACTIONS`,
     - produit un audit totals/rows reutilise dans la fusion.
  9. `liaison-inter-docs` (`component/liaison-inter-docs.py`):
     - detection de liens inter-documents par overlap de topics + matching phrase-a-phrase auditable,
     - ajoute aussi une liaison vectorielle doc-doc et chunk-chunk en reutilisant `ML50_DOC_VECTORS` et `ML50_CHUNK_VECTORS`,
     - audit des meilleurs chunks relies avec similarite cosinus et extraits de texte.
  10. `elasticsearch`
  11. `extraction-regles` (`component/extraction/extraction-regles-50ml.py`):
     - extraction YAML (sans regex de champs) pilotee par classification/doc_type + scoring BM25 par chunk.
  12. `fusion-resultats` (`component/fusion_resultats.py`):
     - fichier unique de fusion, branche `pipeline50ml` incluse dans le meme script,
     - ajout `ml50` + BM25 dans `fusion_output.json`,
     - filtrage des topics grammaticaux (pronoms/determinants/conjonctions/adverbes) via `NLP_TOKENS` du composant grammaire.
- Pipeline `pipeline100ml`:
  1. `pretraitement-de-docs`
  2. `si-image-pretraiter-sinonpass-le-doc`
  3. `output-txt`
  4. `clasification`
  5. `tokenisation-layout` (`component/tokenisation_layout/tokenisation-layout-100ml.py`):
     - tokenisation/layout standard + embeddings Transformer (BERT/XLM-R) avec mean pooling,
     - 1 embedding par chunk + 1 embedding document,
     - topic extraction par chunk (`chunk_primary_topic`, `chunk_topics`) + topics document,
     - generation `NLP_*` minimale (provisoire, puis remplacee par la sortie grammaire),
     - fallback hash local si modele indisponible.
  6. `atripusion-gramatical` (`component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py`):
     - attribution grammaticale XLM-R (`xlm-roberta-base`) multi-langue (FR/EN/AR),
     - sortie `NLP_*` compatible avec le reste de la pipeline,
     - POS 100ml renforce (hybrid rules + contexte + prototypes XLM-R),
     - fallback local automatique si backend Transformer indisponible.
  7. `table-extraction` (`component/table_extraction/table-extraction.py`):
     - extraction tableaux unifiee pour 50ml/100ml (moteur commun),
     - detection agnostique renforcee par geometrie texte (ancrages X stables + score de tabularite par ligne),
     - meilleure detection des tableaux denses/serres OCR,
     - sortie context: `TABLE_EXTRACTIONS_100ML` + `TABLE_EXTRACTIONS`.
  8. `verification-totaux` (`component/verification-totaux.py`):
     - verification non-ML partagee avec `default` et `pipeline50ml`,
     - produit un audit totals/rows reutilise dans `fusion_output.json`.
  9. `detection-signature-chachet-codebarr` (`component/detection-signature-chachet-codebarr.py`):
     - detection visuelle des signatures, cachets, codes-barres et QR codes,
     - moteur hybride: heuristiques visuelles + decodeurs locaux optionnels (`pyzbar`, `OpenCV QRCodeDetector`) si disponibles,
     - sortie context: `VISUAL_MARKS_DETECTIONS_100ML` + `VISUAL_MARKS_DETECTIONS`,
      - localisation normalisee par page (`page_index`, `bbox_px`, `bbox_norm`) et drapeaux document (`has_signature`, `has_stamp`, `has_barcode`, `has_qrcode`).
  10. `liaison-inter-docs` (`component/liaison-inter-docs.py`):
     - detection de liens inter-documents par overlap de topics + matching phrase-a-phrase auditable,
     - ajoute aussi une liaison vectorielle doc-doc et chunk-chunk en reutilisant `ML100_DOC_VECTORS` et `ML100_CHUNK_VECTORS`,
     - audit des meilleurs chunks relies avec similarite cosinus et extraits de texte.
  11. `elasticsearch`
  12. `extraction-regles` (`component/extraction/extraction-regles-100ml.py`):
     - extraction YAML (sans regex de champs) pilotee par classification/doc_type + scoring BM25 par chunk.
  13. `fusion-resultats` (`component/fusion_resultats.py`):
     - fichier unique de fusion, branche `pipeline100ml` incluse dans le meme script,
     - ajout `ml100` + BM25 dans `fusion_output.json`,
     - filtrage des topics grammaticaux (pronoms/determinants/conjonctions/adverbes) via `NLP_TOKENS` du composant grammaire,
     - ajout des sorties visuelles dans `content.visual_flags`, `document_structure.visual_marks`, `document_structure.visual_marks_summary`, `extraction.visual_detection`, `components.detection_signature_chachet_codebarr_100ml` et `pipeline.ml100`.

Selection runtime:
- CLI: `--pipeline default|pipelinorchestrator|pipeline50ml|pipeline100ml`
- variable d'environnement par defaut: `PIPELINE_DEFAULT` (ex: `pipelinorchestrator`, `pipeline50ml` ou `pipeline100ml`; fallback `PIPELINE_PROFILE`)
- etapes CLI: `--only/--upto/--start` utilisent `atripusion-gramatical` (alias legacy accepte: `atripusion-gramatical-en-utilisant-les3ficherla`).

Reference implementation:
- `pipeline/orchestrator.py`

## 4) Flux des donnees (context globals)
- Le pipeline repose sur un `context` dict partage entre composants (exec via `runpy.run_path`).
- Cles majeures produites/consommees:

### 4.1 Entree / pretraitement
- Entree user: `INPUT_FILE`
- Sortie pretraitement type de contenu: `PRETRAITEMENT_RESULT`
- Chaque entree `PRETRAITEMENT_RESULT[]` expose `size` (octets)
- `PRETRAITEMENT_RESULT[].content` peut maintenant valoir:
  - `text`
  - `image_only`
  - `unsupported` (fichier ignore pour OCR)

### 4.2 Routage OCR vs natif
- `TEXT_FILES`
- `IMAGE_ONLY_FILES`
- `UNSUPPORTED_FILES`
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
- en `pipeline50ml`:
  - `ML50_EMBEDDING_METHOD`, `ML50_VECTOR_DIM`
  - `ML50_DOC_VECTORS`, `ML50_CHUNK_VECTORS`, `ML50_WORD_VECTORS`, `ML50_TOPICS`
  - `ML50_CHUNK_VECTORS[]` inclut `chunk_primary_topic` + `chunk_topics`
  - `ML50_TOPICS[]` inclut `document_primary_topics` (top 2 du document) + `document_topics`
  - topic extractor ameliore: filtrage bruit OCR + n-grams + scoring TF-IDF pondere + boost classification
  - `NLP_ANALYSES`/`NLP_TOKENS` provisoires construits par tokenisation 50ml (puis remplaces par la sortie grammaire dans la pipeline complete)
- en `pipeline100ml`:
  - `ML100_EMBEDDING_METHOD`, `ML100_EMBEDDING_BACKEND`, `ML100_MODEL_NAME`, `ML100_VECTOR_DIM`
  - `ML100_DOC_VECTORS`, `ML100_CHUNK_VECTORS`, `ML100_WORD_VECTORS`, `ML100_TOPICS`
  - `ML100_CHUNK_VECTORS[]` inclut `chunk_primary_topic` + `chunk_topics`
  - `ML100_TOPICS[]` inclut `document_primary_topics` (top 2 du document) + `document_topics`
  - topic extractor ameliore: filtrage bruit OCR + n-grams + scoring TF-IDF pondere + boost classification
  - `NLP_ANALYSES`/`NLP_TOKENS` provisoires construits par tokenisation 100ml (puis remplaces par la sortie grammaire dans la pipeline complete)

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

### 4.6 Extraction de tableaux (pipelines 50ml/100ml)
- composant runtime unique: `component/table_extraction/table-extraction.py`
- moteur commun: `component/table_extraction/table_extraction_lib.py`
- sorties contexte:
  - `TABLE_EXTRACTIONS` (alias commun),
  - `TABLE_EXTRACTIONS_50ML` (pipeline50ml),
  - `TABLE_EXTRACTIONS_100ML` (pipeline100ml).
- objectifs:
  - detecter tableaux espaces et tableaux OCR serres/compacts,
  - fallback OCR multi-variants (`psm 3/4/6`, images autocontrast/threshold/sharpen/upscale) pour reconstruire des lignes/cellules quand la sortie locale est incomplète,
  - extraire `reference`, `product`, `quantity`, `unit_price`, `total`, `total_ht`, `total_ttc`, `tax` + `raw_cells`,
  - structurer chaque tableau avec `table_type` + `shape` (`columns_estimated`, `rows_estimated`),
  - supprimer automatiquement les tableaux redondants/partiels quand un tableau riche est detecte.

### 4.6 bis Verification des totaux (3 pipelines)
- composant runtime: `component/verification-totaux.py`
- sortie contexte:
  - `TOTALS_VERIFICATION`
- verification effectuee:
  - coherence ligne par ligne (`quantity * unit_price ~= total`)
  - sous-total recompose depuis les lignes
  - comparaison aux montants declares extraits du tableau (`total_ht`, `tax`, `total`, `total_ttc`, `amount_due`)
- sortie par document:
  - `verification_status` (`ok`, `partial_ok`, `mismatch`, `not_enough_data`)
  - `passed`, `complete`
  - `computed_subtotal`, `declared_subtotal`, `declared_tax`, `declared_total`, `expected_total`
  - `row_audit[]` avec detail des lignes verifiees
  - audit de localisation:
    - `table_anchor`
    - `subtotal_location`, `tax_location`, `total_location`
    - `issue_locations[]`
    - chaque localisation peut exposer `table_index`, `page_index`, `sent_index`, `line`, `chunk_start`, `chunk_end`

### 4.7 Detection visuelle signature/cachet/code-barres/QR (pipeline100ml)
- composant runtime: `component/detection-signature-chachet-codebarr.py`
- sorties contexte:
  - `VISUAL_MARKS_DETECTIONS_100ML`
  - `VISUAL_MARKS_DETECTIONS` (alias commun)
- contenu par document:
  - `engine`, `source_path`, `pages_scanned`, `detections_count`
  - `has_signature`, `has_stamp`, `has_barcode`, `has_qrcode`
  - `detections[]` avec `page_index`, `bbox_px`, `bbox_norm`, `score`, `source`
- objectif:
  - marquer rapidement la presence d'une signature, d'un cachet, d'un code-barres ou d'un QR code,
  - exposer leur emplacement dans le document pour audit/fusion,
  - utiliser les decodeurs locaux quand disponibles, sinon fallback heuristique.

### 4.8 Liaison inter-documents
- composant: `component/liaison-inter-docs.py`
- sortie contexte:
  - `INTERDOC_ANALYSIS` (methode, statistiques, liens)
  - `INTERDOC_LINKS` (liens inter-documents)
  - `INTERDOC_DOC_LINKS` (index doc -> link_ids)
- audit phrase-a-phrase:
  - chaque lien contient `audit.matches[]` avec `phrase_a` / `phrase_b` (page/sentence/text_excerpt), `shared_terms`, `shared_topics`, `score`.
  - `shared_terms` est nettoye pour garder des termes informatifs (stopwords/pronoms/mots vides exclus; priorite aux termes semantiques POS/lemma + topics + signaux classification).

### 4.9 Elasticsearch (optionnel)
- activation: `USE_ELASTICSEARCH`
- conf: `ES_URL`, `ES_INDEX`
- auth HTTP: `ES_USER`, `ES_PASSWORD`, `ES_API_KEY`
- conf NLP: `ES_NLP_LEVEL` (`off|summary|full`), `ES_NLP_INDEX`, `ES_NLP_MAX_FULL_TOKENS`
- auto-start: `ES_AUTO_START`, `ES_START_COMMAND`, `ES_START_COMMANDS`, `ES_START_COMMAND_POSIX`, `ES_START_COMMAND_WINDOWS`, `ES_START_PASSWORD`, `ES_AUTO_START_WAIT_SECONDS`, `ES_AUTO_START_LAUNCH_TIMEOUT`
  - comportement par defaut: auto-start Windows uniquement, Linux/macOS en demarrage manuel.
- sorties: `ES_AVAILABLE`, `ES_DOC_IDS`, `ES_CLASSIFICATION_DOCS`, `ES_EXTRACTION_DOCS`, `ES_AUTO_STARTED`, `ES_AUTO_START_CMD`
- sorties NLP ES: `ES_NLP_SYNC`, `ES_NLP_DOCS_SYNCED`, `ES_NLP_TOKENS_SYNCED`, `ES_NLP_TOKEN_ERRORS`, `ES_NLP_LEVEL_EFFECTIVE`, `ES_NLP_INDEX_EFFECTIVE`

### 4.10 Classification
- sortie: `RESULTS` (doc_type, status, scores, `classification_log`, `keyword_matches`, `anti_confusion_targets`)
- sync ES: `ES_CLASSIFICATION_SYNCED`

### 4.11 Extraction metier
- sortie: `EXTRACTIONS`
- sync ES: `ES_EXTRACTION_SYNCED`
- en `pipeline50ml`: extraction basee YAML (`rules/*.yaml`, `config/ruleset_routes.yaml`) + `EXTRACTIONS[].bm25` + `BM25_RESULTS`
- en `pipeline100ml`: extraction basee YAML (`rules/*.yaml`, `config/ruleset_routes.yaml`) + `EXTRACTIONS[].bm25` + `BM25_RESULTS`

### 4.12 Fusion finale
- sortie fichier: `fusion_output.json`
- flags contexte: `FUSION_RESULT`, `FUSION_PAYLOAD`, `FUSION_PAYLOADS`, `FUSION_SOURCE`, `ES_FUSION_SYNCED`
- structure finale: `documents[]` (un bloc complet par document, avec `components`, `text`, `document_structure`, `extraction`, `nlp`, `quality_checks`)
- en source `local-context`, la fusion est multi-doc (1 payload par document detecte dans `TOK_DOCS`/`FINAL_DOCS`) au lieu d'un seul document.
- structure finale ajoutee: `cross_document_analysis` (liens inter-documents + audit des phrases match)
- chaque document ajoute `cross_document` (`linked_documents_count`, `link_ids`) sans dupliquer l'audit complet.
- en `pipeline50ml`: bloc supplementaire `document.ml50`, `pipeline.profile="pipeline50ml"` et `pipeline.ml50`
  - `document.ml50.document_primary_topics`: 2 topics principaux du document
- en `pipeline100ml`: bloc supplementaire `document.ml100`, `pipeline.profile="pipeline100ml"` et `pipeline.ml100`
  - `document.ml100.document_primary_topics`: 2 topics principaux du document
  - `content.visual_flags`: drapeaux visuels documentaires
  - `document_structure.visual_marks`: localisations detectees
  - `document_structure.visual_marks_summary`: resume detection visuelle
  - tous profils:
  - `extraction.totals_verification`
  - `components.verification_totaux`
  - `quality_checks[]` inclut un check `totals_verification`
  - les audits de verification des totaux exposent aussi l'emplacement precis de l'erreur/manque dans le tableau et le chunk source

## 5) Ou modifier selon le besoin

### 5.1 Changer la CLI, options, sequence des etapes
- `pipeline/cli.py`
- default pipeline hardcodee modifiable dans le code: `PIPELINE_DEFAULT_CODE`
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
- `component/tokenisation_layout/tokenisation-layout.py`
  - `layout_items`
  - `_transpose_or_group_multicol`
  - `_collect_table_block`
  - `chunk_layout_universal`
  - `chunk_is_noise`
- variante ML50 (embedding/topic/doc-vector):
  - `component/tokenisation_layout/tokenisation-layout-50ml.py`
- variante ML100 (Transformer BERT/XLM-R + pooling):
  - `component/tokenisation_layout/tokenisation-layout-100ml.py`
- extraction de tableaux (pipelines 50ml/100ml):
  - `component/table_extraction/table-extraction.py` (script runtime unique),
  - `component/table_extraction/table_extraction_lib.py` (moteur commun et heuristiques).

### 5.6 Changer la liaison inter-documents (topics + audit phrases)
- `component/liaison-inter-docs.py`
  - calcul des liens entre documents,
  - regles de score/threshold,
  - structure de l'audit `phrase_a` / `phrase_b` et champs exposes dans la fusion.

### 5.6 bis Changer la verification des totaux
- `component/verification-totaux.py`
  - parsing des montants/quantites,
  - choix des candidats de totaux les plus plausibles,
  - audit ligne par ligne,
  - regles de statut `ok` / `partial_ok` / `mismatch` / `not_enough_data`.

### 5.6 ter Changer la detection visuelle signature/cachet/code-barres/QR
- `component/detection-signature-chachet-codebarr.py`
  - chargement des pages source (images/PDF),
  - heuristiques visuelles signature/cachet,
  - decodeurs QR/code-barres optionnels,
  - seuils de score et structure de sortie.

### 5.7 Changer classification documentaire (scores, threshold, priorites)
- code: `component/clasification.py`
- config: `classification/common.json`, `classification/*.json`

### 5.8 Changer extraction metier
- moteur regex (pipeline default): `component/extraction/extraction-regles.py`
- moteur YAML (pipelines 50ml/100ml): `component/extraction/extraction-regles-yaml.py`
- routage rulesets regex: `config/ruleset_routes.json`
- routage rulesets YAML: `config/ruleset_routes.yaml`
- patterns metier regex: `rules/*.json`
- patterns metier YAML: `rules/*.yaml`
- variante ML50 avec BM25:
  - `component/extraction/extraction-regles-50ml.py`
- variante ML100 avec BM25:
  - `component/extraction/extraction-regles-100ml.py`

### 5.9 Changer logique Elasticsearch
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

### 5.10 Changer fusion JSON finale
- `component/fusion_resultats.py`
- fichier unique de fusion pour `default`, `pipeline50ml`, `pipeline100ml` (pilotage par `PIPELINE_PROFILE`)
- en mode `NLP full`, la fusion charge aussi les tokens NLP depuis ES (`dms_nlp_tokens`) pour le document courant et les structure par page/sentence.
- schema fusion lisible humain + exploitable code:
  - top-level: `schema_version`, `generated_at`, `source`, `documents_count`, `documents`
  - par document: `components` detaille tous les composants du pipeline
- classification detaillee exposee (`scores`, `classification_log`, `keyword_matches`, `anti_confusion_targets`) + `file.size`
- deduplication active: `components` reste volontairement compact (resume) et les details complets restent au niveau document (`classification`, `extraction`, `nlp`, `file`)
- branche `pipeline50ml`:
  - enrichit la fusion avec `ml50` (vectors/topics) + BM25
- branche `pipeline100ml`:
  - enrichit la fusion avec `ml100` (vectors/topics) + BM25

### 5.11 Changer logique linguistique EN/FR/AR
- orchestrateur langues: `component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py`
- variante dediee pipeline100 (XLM-R): `component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py`
- anglais: `component/atrribution-gramatical/engcode.py`
- francais: `component/atrribution-gramatical/frcode.py`
- arabe: `component/atrribution-gramatical/arabcode.py`

### 5.12 Telechargements automatiques (global)
- 5.12.1 Grammaire pipeline100 (XLM-R)
  - composant:
    - `component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py`
  - modele par defaut:
    - `xlm-roberta-base`
  - artefacts telecharges automatiquement si absents:
    - `config.json`
    - `tokenizer_config.json`
    - `tokenizer.json`
    - `special_tokens_map.json`
    - `sentencepiece.bpe.model`
    - poids (`model.safetensors` ou `pytorch_model.bin`)
  - ou les trouver:
    1. `ML100_MODEL_LOCAL_DIR` si defini (pas de download)
    2. cache projet (`ML100_MODEL_CACHE_DIR` ou defaut):
       - `/home/mourad/Bureau/DMS/core/component/atrribution-gramatical/.hf_model_cache`
    3. fallback `transformers` direct Hub:
       - `~/.cache/huggingface/hub` (ou `HF_HOME` / `TRANSFORMERS_CACHE`)
  - variables:
    - `ML100_MODEL_NAME`, `ML100_MODEL_LOCAL_DIR`, `ML100_MODEL_CACHE_DIR`
    - offline: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `LANG_PIPE_OFFLINE=1`
  - audit terminal:
    - `[grammar-100ml-xlmr] ... source=...`
    - `[grammar-100ml-xlmr][model] auto-installed:... | remote-hub:... | local-dir:...`

- 5.12.2 Embeddings pipeline100 (tokenisation)
  - composant:
    - `component/tokenisation_layout/tokenisation-layout-100ml.py`
  - telechargement possible:
    - modele `ML100_MODEL_NAME` via `AutoTokenizer.from_pretrained` + `AutoModel.from_pretrained`
  - ou les trouver:
    - `~/.cache/huggingface/hub` (ou `HF_HOME` / `TRANSFORMERS_CACHE`)

- 5.12.3 Grammaire EN/FR (pipeline default + pipeline50ml)
  - composants:
    - `component/atrribution-gramatical/engcode.py`
    - `component/atrribution-gramatical/frcode.py`
  - telechargements automatiques:
    - NLTK (EN): `punkt`, `averaged_perceptron_tagger`, `wordnet`, `omw-1.4`
    - modele NER EN: `dslim/bert-base-NER`
    - modele NER FR: `Davlan/bert-base-multilingual-cased-ner-hrl`
  - ou les trouver:
    - NLTK: `~/nltk_data` (ou `NLTK_DATA`)
    - HF models: `~/.cache/huggingface/hub` (ou `HF_HOME` / `TRANSFORMERS_CACHE`)

- 5.12.4 Tokenisation layout classique
  - composant:
    - `component/tokenisation_layout/tokenisation-layout.py`
  - telechargements automatiques:
    - NLTK: `punkt`, `punkt_tab`
  - ou les trouver:
    - `~/nltk_data` (ou `NLTK_DATA`)

- 5.12.5 Non auto-installe (manuel requis)
  - `tesseract` (OCR systeme): non telecharge/installe automatiquement
  - `camel_tools` + donnees (`morphology-db-msa-r13`, `ner-arabert`): commandes fournies, installation manuelle

## 6) Cartographie des fichiers Python (roles)

### 6.1 Orchestration (`pipeline/`)
- `pipeline/settings.py`: logging, normalize input, context managers cwd/argv, chargement `.env` optionnel.
- `pipeline/components.py`: wrappers des composants scripts + resumes + sync ES.
- `pipeline/orchestrator.py`: contient les 3 orchestrateurs (`PipelineOrchestrator`, `Pipeline50MLOrchestrator`, `Pipeline100MLOrchestrator`) + selection `only/upto/start`.
- `pipeline/cli.py`: CLI + chargement `.env` + tee print vers `outputgeneralterminal.txt`.
- `pipeline/elasticsearch.py`: store HTTP ES + auth + flatten/index + auto-start local ES (POSIX/Windows) + fallback docs + sync classification/extraction + sync NLP (summary/full).
- `pytesseract.py`: shim local compatible `pytesseract` base sur le binaire `tesseract` (OCR CLI, OSD, TSV, langues, version) quand le package Python n'est pas installe.

### 6.2 Composants metier (`component/`)
- `pretraitement-de-docs.py`: detect format, determine `text` vs `image_only`, extrait `size`.
- `si-image-pretraiter-sinonpass-le-doc.py`: split OCR/native + preprocess image.
- `output-txt.py`: OCR tesseract + extraction native multi-format + `FINAL_DOCS` (+ `size`).
- `tokenisation_layout/tokenisation-layout.py`: language detect + sentence/layout chunking + table/multicol + `TOK_DOCS` (+ `size`).
- `tokenisation_layout/tokenisation-layout-50ml.py`: tokenisation/layout + embeddings FastText-like + topics + vectors doc/chunk/word + `NLP_*` minimal (provisoire avant grammaire).
  - topics chunk-level (`chunk_primary_topic`, `chunk_topics`) + top-2 document (`document_primary_topics`).
  - scoring topics ameliore (n-grams, dedupe semantique, boost keywords classification, anti-bruit OCR).
- `tokenisation_layout/tokenisation-layout-100ml.py`: tokenisation/layout + embeddings Transformer (BERT/XLM-R) + pooling mean + vectors chunk/doc + topics + `NLP_*` minimal (provisoire avant grammaire).
  - fallback hash si modele non disponible localement.
  - scoring topics ameliore (n-grams, dedupe semantique, boost keywords classification, anti-bruit OCR).
- `atrribution-gramatical/*.py`: POS/lemma/NER per language + notebook style runners.
- `atrribution-gramatical/attribution-gramatical-100ml-xlmr.py`: composant grammaire dedie `pipeline100ml`, base XLM-R (`xlm-roberta-base`) FR/EN/AR, sortie compatible `NLP_*` + fallback local.
- `liaison-inter-docs.py`: lie les documents entre eux via topics + recouvrement lexical phrase-a-phrase, puis publie un audit des matches.
  - filtre qualite `shared_terms`: supprime mots non-significatifs (`son`, `tout`, etc.) et favorise termes metier/juridiques.
- `elasticsearch.py`: step script pour index/fetch docs ES.
- `clasification.py`: keyword scoring classification + details matches (`classification_log`, `keyword_matches`).
- `extraction/extraction-regles.py`: regex extractors selon doc_type.
- `extraction/extraction-regles-yaml.py`: extraction sans regex de champs (labels/detecteurs) selon doc_type via YAML.
- `extraction/extraction-regles-50ml.py`: extraction-regles-yaml + scoring BM25 sur chunks.
- `extraction/extraction-regles-100ml.py`: extraction-regles-yaml + scoring BM25 sur chunks.
- `table_extraction/table-extraction.py`: composant unique d'extraction tableaux pour pipeline50ml + pipeline100ml.
- `table_extraction/table_extraction_lib.py`: logique commune (split lignes OCR denses, detection header, mapping colonnes, extraction line-items, fallback OCR multi-variants, dedup/pruning de tableaux redondants).
- `fusion_resultats.py`: build JSON fusion structure par document (`documents[]`) depuis context + ES.
  - ajoute `cross_document_analysis` (liens + audit) et `document.cross_document` (references link_ids).
- `fusion_resultats.py` contient aussi les branches profilees:
  - `pipeline50ml`: enrichissement `document.ml50`, `components.tokenisation_layout_50ml`, `components.extraction_regles_50ml`, `components.table_extraction_50ml`, BM25, filtrage grammatical des topics, print `[ml50-topic]`.
  - `pipeline100ml`: enrichissement `document.ml100`, `components.tokenisation_layout_100ml`, `components.extraction_regles_100ml`, `components.table_extraction_100ml`, BM25, filtrage grammatical des topics, print `[ml100-topic]`.

## 7) Fichiers metier JSON/YAML
- `classification/common.json`: poids/penalites/seuil/marge globaux.
- `classification/*.json`: classes documentaires + keywords + anti-confusion.
- `config/ruleset_routes.json`: mapping doc_type -> rulesets.
- `config/ruleset_routes.yaml`: mapping doc_type -> rulesets YAML (utilise par pipelines 50ml/100ml).
- `rules/common.json`: extracteurs communs (date/email/phone/url).
- `rules/FACTURE.json`: extracteurs facture FR/EN/AR.
- `rules/BON_DE_COMMANDE.json`: extracteurs BC FR/EN/AR.
- `rules/CONTRAT.json`: extracteurs contrat FR/EN/AR.
- `rules/common.yaml`: extracteurs communs YAML (detectors simples).
- `rules/FACTURE.yaml`: extracteurs facture YAML (labels metier).
- `rules/BON_DE_COMMANDE.yaml`: extracteurs BC YAML (labels metier).
- `rules/CONTRAT.yaml`: extracteurs contrat YAML (labels metier).

## 8) Artefacts d'execution
- `orchestre.log`: log Python (logging)
- `outputgeneralterminal.txt`: tee des `print(...)`
- `fusion_output.json`: sortie fusion finale

## 9) Notes de qualite observees pendant audit
- `component/si-image-pretraiter-sinonpass-le-doc.py` contient du code duplique sur la construction de `DOCS` (deux boucles consecutives).
- `component/extraction/extraction-regles.py` compile les regex sans garde plus fine que `re.error`; en cas de pattern invalide, le champ est skippe (comportement tolerant).
- `component/fusion_resultats.py` est clairement oriente "debug/fusion", pas schema strict valide via validation formelle.
- La couche `pipeline/` est proprement separee et sert de facade stable autour des scripts notebooks.

## 10) Index complet des fonctions/classes
- Voir `FUNCTION_INDEX.txt` pour la liste exhaustive `file:line:def/class`.
- Ce fichier est la reference la plus rapide pour localiser une modification precise.

## 11) Changelog code
- 2026-03-29:
  - `prompt_output.json`:
    - sortie JSON stricte regeneree selon la version courante de `prompt.txt`.
    - aligne le schema final unifie sur les cles racine demandees:
      - `schema_version`
      - `generated_at`
      - `source`
      - `profile`
      - `null_policy`
      - `documents_count`
      - `documents`
      - `cross_document_analysis`
      - `pipeline`
      - `registries`
      - `item_templates`
      - `sql_mapping_hints`
    - conserve un seul objet JSON racine, avec blocs `default` / `pipeline50ml` / `pipeline100ml` unifies dans le meme template.
    - initialise les scalaires absents a `null`, les tableaux a `[]` et laisse les objets presents dans la structure finale.
    - complete avec les champs reels observes dans `fusion_output.json` et les enrichissements statiques de `component/fusion_resultats.py`.
    - couvre maintenant aussi:
      - `content.content_type` / `content.classification` / `content.document_kind` / `content.detected_languages`
      - `text.search.title` / `text.search.keywords`
      - `nlp.source` / `nlp.level` / `nlp.summary` / `nlp.full`
      - `classification.scores_audit.*` et `classification.keyword_matches.*`
      - `extraction.regex_extractions.fields.*` via template dynamique
      - `extraction.table_extraction.*`
      - `extraction.totals_verification.*`
      - `cross_document_analysis.links[*]` avec audit phrase et audit vectoriel
    - verification de couverture effectuee automatiquement: aucun chemin present dans `fusion_output.json` n'est absent du template final.
  - `GLOBAL_UNIFIED_PIPELINE_OUTPUT_TEMPLATE.json`:
    - nouveau contrat JSON global de sortie couvrant `default`, `pipeline50ml` et `pipeline100ml`.
    - inclut tous les champs observes dans les 3 profils.
    - les champs specifiques a un profil restent presents et prennent `null` si le pipeline courant ne les alimente pas.
    - inclut aussi des `item_templates` pour les structures listees (`interdoc_link`, `chunk_embedding_item`, `table_item`, `regex_extraction_item`, `visual_mark_item`, etc.).
  - `main.py`:
    - ajoute un shebang Unix pour permettre `./main.py <fichier> ...`.
  - `run-dms`:
    - nouveau lanceur shell simple qui appelle automatiquement `main.py` avec le Python actif du venv si disponible.
    - evite la confusion ou le document est tape comme une commande shell.
  - `README.md`:
    - exemples corriges:
      - `./main.py <fichier>`
      - `./run-dms <fichier>`
    - ajoute un exemple explicite "faux" vs "correct" pour l'erreur `Permission non accordee`.
  - `component/pretraitement-de-docs.py`:
    - ne lance plus de faux "test" au chargement du script.
    - expose maintenant `MISSING_FILES` quand des chemins d'entree sont invalides/introuvables.
  - `component/si-image-pretraiter-sinonpass-le-doc.py`:
    - suppression du doublon de construction `DOCS`, qui causait des doubles logs `Using INPUT_FILE=...`.
    - `UNSUPPORTED_FILES` reste reserve aux fichiers existants mais non gerables par OCR.
    - les fichiers introuvables remontent proprement via `MISSING_FILES`.
  - `pipeline/components.py`:
    - `pretraitement-de-docs` et `si-image-pretraiter-sinonpass-le-doc` reportent maintenant `missing=...`.
    - fusionne les `MISSING_FILES` amont/aval pour eviter de perdre l'information entre les etapes.
  - `component/detection-signature-chachet-codebarr.py`:
    - corrige le blocage sur gros PDF (> 200 pages) visible dans `bug.txt`.
    - le rendu PDF ne lance plus `pdftoppm` sur tout le document puis ne garde que les premieres pages.
    - ajout de `_pdf_page_count()` + `_sample_page_numbers()`:
      - debut du document
      - pages intermediaires reparties
      - fin du document
    - rendu page par page avec `pdftoppm -f N -l N -singlefile`, timeout borne et DPI reduit.
    - sortie enrichie:
      - `pages_total`
      - `pages_scanned`
      - `sampled_pages`
    - print terminal enrichi: `pages=scannees/total` + apercu des pages echantillonnees.
  - `component/liaison-inter-docs.py`:
    - corrige un blocage de performance sur les comparaisons vectorielles chunk-a-chunk en `pipeline50ml` / `pipeline100ml`.
    - les vecteurs document/chunk sont normalises une seule fois puis compares via produit scalaire.
    - ajoute un budget de calcul par lien:
      - reduction des chunks candidats (`max_chunk_candidates`)
      - plafond de paires vectorielles (`max_chunk_pairs`)
      - skip des paires sans recouvrement lexical/topic quand la similarite document reste faible.
    - l'audit expose maintenant:
      - `chunk_pair_budget_applied`
      - `candidate_chunks_a`
      - `candidate_chunks_b`
      - `original_chunks_a`
      - `original_chunks_b`
      - `pairs_skipped_without_overlap`
  - `component/pretraitement-de-docs.py`:
    - les fichiers non supportes (ex: `Unknown / binary`, `application/octet-stream`) sont maintenant etiquetes `unsupported` au lieu d'etre envoyes vers le flux OCR.
    - les fichiers `.txt` sont reconnus comme `text`.
  - `component/si-image-pretraiter-sinonpass-le-doc.py`:
    - ajoute `UNSUPPORTED_FILES`.
    - les fichiers `unsupported` sont explicitement ignores avec logs:
      - `[skip] content='unsupported' -> ...`
      - `[unsupported] fichiers ignores (non geres par OCR):`
    - evite le crash `PIL.UnidentifiedImageError` quand un `.txt` ou un binaire non image se retrouve dans l'entree.
    - les fichiers `.txt` passent dans le flux `text` et ne sont plus envoye a OCR.
  - `component/output-txt.py`:
    - ajoute l'extraction native des `.txt` avec fallback d'encodage `utf-8` -> `utf-8-sig` -> `latin-1`.
    - sortie `extraction`: `native:txt:utf-8`, `native:txt:utf-8-sig` ou `native:txt:latin-1`.
  - `pipeline/cli.py`:
    - si `outputgeneralterminal.txt` est fourni comme document d'entree, les logs runtime sont rediriges vers `outputgeneralterminal.runtime.txt` pour ne pas ecraser le document source pendant l'analyse.
  - `pipeline/components.py`:
    - reporting enrichi:
      - `pretraitement-de-docs`: compte `unsupported`
      - `si-image-pretraiter-sinonpass-le-doc`: expose aussi `UNSUPPORTED_FILES` dans `PREPROCESS_RESULT`.
- 2026-03-28:
  - `component/verification-totaux.py`:
    - nouveau composant non-ML partage par `default`, `pipeline50ml` et `pipeline100ml`.
    - verifie les lignes (`quantity * unit_price`), reconstruit un sous-total, choisit les totaux declares les plus plausibles et calcule un statut `ok` / `partial_ok` / `mismatch`.
    - ajoute maintenant une auditabilite fine des erreurs/manques:
      - `table_index`
      - `page_index`
      - `sent_index` / chunk source
      - offsets `chunk_start` / `chunk_end`
      - `issue_locations[]`, `table_anchor`, `subtotal_location`, `tax_location`, `total_location`.
  - `pipeline/orchestrator.py`:
    - ajoute `verification-totaux` juste apres `table-extraction` dans les 3 pipelines.
  - `pipeline/components.py`:
    - ajoute `TotalsVerificationComponent` avec resume terminal standardise.
  - `pipeline/cli.py`:
    - ajoute `verification-totaux` aux etapes supportees par `--only`, `--upto`, `--start`.
  - `component/fusion_resultats.py`:
    - injecte l'audit de verification dans:
      - `extraction.totals_verification`
      - `components.verification_totaux`
      - `quality_checks[]`
      - compteurs `pipeline.0ml`, `pipeline.ml50`, `pipeline.ml100`.
    - propage aussi les localisations precises d'erreur/manque pour faciliter l'audit humain.
- 2026-03-28:
  - `component/detection-signature-chachet-codebarr.py`:
    - nouveau composant pipeline100ml place entre `table-extraction` et `liaison-inter-docs`.
    - detecte signatures, cachets, codes-barres et QR codes avec localisation par page.
    - corrige la resolution des chemins source en supportant les chemins relatifs issus de `PRETRAITEMENT_RESULT`.
    - ajoute des decodeurs locaux optionnels (`pyzbar`, `OpenCV QRCodeDetector`) pour ameliorer la precision QR/code-barres quand disponibles.
  - `pipeline/orchestrator.py`:
    - ajoute l'etape `detection-signature-chachet-codebarr` dans `Pipeline100MLOrchestrator`.
  - `pipeline/components.py`:
    - ajoute le wrapper `VisualMarksDetectionComponent` avec resume terminal standardise.
  - `pipeline/cli.py`:
    - ajoute `detection-signature-chachet-codebarr` dans les etapes supportees par `--only`, `--upto`, `--start`.
  - `component/fusion_resultats.py`:
    - integre les detections visuelles 100ml dans `fusion_output.json`:
      - `content.visual_flags`
      - `document_structure.visual_marks`
      - `document_structure.visual_marks_summary`
      - `extraction.visual_detection`
      - `components.detection_signature_chachet_codebarr_100ml`
      - compteurs `pipeline.ml100.visual_*`.
- 2026-03-19:
  - `EXPLICATION_PIPELINES.txt`:
    - nouveau fichier texte de lecture rapide pour les 3 pipelines.
    - explique la selection runtime via `PIPELINE_DEFAULT_CODE` dans `pipeline/cli.py`.
    - liste les `self.components` exacts de `PipelineOrchestrator`, `Pipeline50MLOrchestrator` et `Pipeline100MLOrchestrator`.
    - resume le chainage des composants et les informations extraites par chaque etape.
    - precise aussi les technos/moteurs utilises par composant: `tesseract`, `simplemma`, `WordNet`, `camel_tools`, BERT NER legacy, `xlm-roberta-base`, BM25 maison, et les non-usages comme `Word2Vec`.
  - `component/si-image-pretraiter-sinonpass-le-doc.py`:
    - l'import `IPython.display` est maintenant optionnel.
    - en execution CLI, le composant n'essaie plus d'afficher les images notebook et ne plante plus si `IPython` est absent.
  - `pytesseract.py`:
    - ajout d'un shim local compatible avec les usages du projet:
      - `get_tesseract_version`
      - `get_languages`
      - `image_to_osd`
      - `image_to_string`
      - `image_to_data`
      - `Output.DICT`
    - le shim s'appuie sur le binaire systeme `tesseract` deja present sur la machine et evite la dependance obligatoire au package Python `pytesseract`.
    - effet: suppression du crash `ModuleNotFoundError: No module named 'pytesseract'` sur les etapes OCR et extraction tableaux, tant que le binaire `tesseract` est disponible.
  - `component/tokenisation_layout/tokenisation-layout.py`:
    - `nltk` est maintenant optionnel.
    - fallback local de segmentation phrase/layout si `nltk` ou les donnees `punkt` sont absents.
  - `component/tokenisation_layout/tokenisation-layout-100ml.py`:
    - suppression de la dependance dure a `numpy`.
    - calcul des vecteurs hash/mean et des sorties 100ml en listes Python pures.
  - `component/tokenisation_layout/tokenisation-layout-50ml.py`:
    - suppression de la dependance dure a `numpy`.
    - calcul des vecteurs FastText-like/hash et moyennes en listes Python pures.
  - `pipeline/orchestrator.py`:
    - reintegre `table-extraction` dans la pipeline `default` apres `atripusion-gramatical`.
  - `component/table_extraction/table-extraction.py`:
    - distingue maintenant le profil `0ml`/`default` au lieu de se faire passer pour `100ml`.
    - expose `TABLE_EXTRACTIONS_DEFAULT` pour la branche standard.
  - `component/table_extraction/table_extraction_lib.py`:
    - en profil non-ML, renseigne `TABLE_EXTRACTIONS_DEFAULT` + `TABLE_EXTRACTIONS`.
    - engine runtime maintenant tagge `table-0ml-unified-v3-anchor-geometry`.
  - `component/fusion_resultats.py`:
    - ajoute une integration specifique `0ml` pour les tableaux detectes dans la pipeline standard.
    - injecte `extraction.table_extraction`, `components.table_extraction_0ml`, `pipeline.0ml` et alimente `document_structure.tables`.
    - ajoute les prints terminal `[table-0ml]` et `[fusion-resultats-0ml]`.
  - `component/liaison-inter-docs.py`:
    - ajoute une couche de liaison vectorielle active seulement en `pipeline50ml` et `pipeline100ml`.
    - `pipeline50ml`: reutilise `ML50_DOC_VECTORS` et `ML50_CHUNK_VECTORS` (FastText-like local).
    - `pipeline100ml`: reutilise `ML100_DOC_VECTORS` et `ML100_CHUNK_VECTORS` (embeddings Transformer/XLM-R).
    - calcule une similarite doc-doc + des meilleurs couples chunk-chunk avec audit de texte, score hybride et similarite cosinus.
    - enrichit `INTERDOC_ANALYSIS` avec `vector_profile`, `chunk_pairs_scored`, `vector_links_count` et `vector_audit` par lien.
  - `pipeline/components.py`:
    - resume du composant `liaison-inter-docs` enrichi avec `chunk_pairs_scored`, `vector_profile` et `vector_links_count`.
  - `component/fusion_resultats.py`:
    - exporte maintenant les metadonnees vectorielles inter-docs dans `cross_document_analysis`.
  - `component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py`:
    - suppression de la dependance dure a `numpy`.
    - encodeur/fallback XLM-R et raffinement POS vectoriel adaptes a des matrices/listes Python pures.
  - architecture `component/`:
    - creation du sous-dossier `component/extraction/` pour centraliser:
      - `extraction-regles.py`
      - `extraction-regles-yaml.py`
      - `extraction-regles-50ml.py`
      - `extraction-regles-100ml.py`
    - creation du sous-dossier `component/tokenisation_layout/` pour centraliser:
      - `tokenisation-layout.py`
      - `tokenisation-layout-50ml.py`
      - `tokenisation-layout-100ml.py`
    - ajout des marqueurs package:
      - `component/extraction/__init__.py`
      - `component/tokenisation_layout/__init__.py`
  - `pipeline/orchestrator.py`:
    - mise a jour de tous les chemins scripts vers la nouvelle arborescence `component/extraction/` et `component/tokenisation_layout/`.
    - `fusion-resultats` unifie pour `default`, `pipeline50ml` et `pipeline100ml` via le meme script `component/fusion_resultats.py`.
  - `component/fusion_resultats.py`:
    - suppression du besoin des wrappers `fusion_resultats-50ml.py` et `fusion_resultats-100ml.py`.
    - consolidation des branches profilees dans le meme fichier via `PIPELINE_PROFILE`.
    - separation explicite entre base commune et enrichissements profiles pour faciliter la maintenance.
  - `component/extraction/extraction-regles.py` + `component/extraction/extraction-regles-yaml.py`:
    - correction du calcul `REPO_ROOT` apres deplacement en sous-dossier (`parents[2]`).
  - `graphecode.html`:
    - remise a jour du graphe runtime avec le bon ordre d'execution actuel:
      - `clasification` avant `tokenisation-layout`,
      - ajout de `liaison-inter-docs`,
      - chemins scripts corriges vers les nouveaux sous-dossiers.
  - validation technique:
    - compilation `py_compile` OK sur `pipeline/orchestrator.py`, `component/fusion_resultats.py`, les scripts de `component/extraction/` et `component/tokenisation_layout/`.
    - validation fonctionnelle locale du moteur d'extraction et de la fusion unifiee via contexte synthetique.
    - validation runtime reelle:
      - `python main.py documents/image2tab.webp --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens`
      - resultat: execution complete jusqu'a `fusion-resultats` avec `EXIT:0`.
- 2026-03-18:
  - `.gitignore`:
    - ajout des ignores pour caches de telechargement modeles/transformers afin d'eviter les commits de poids inutiles:
      - `component/atrribution-gramatical/.hf_model_cache/`
      - `.hf_model_cache/`
      - `.cache/huggingface/`
      - `.huggingface/`
  - `pipeline/orchestrator.py`:
    - renommage du nom d'etape runtime en `atripusion-gramatical` (au lieu du nom legacy long) pour les 3 pipelines.
  - `pipeline/components.py`:
    - audit explicite du composant grammaire utilise via `NLP_GRAMMAR_COMPONENT_NAME` et `NLP_GRAMMAR_COMPONENT_SCRIPT`.
    - resume terminal enrichi (`backend=... | component=atripusion-gramatical`) pour confirmer l'implementation active.
  - `component/fusion_resultats.py`:
    - bloc `components.attribution_grammaticale` enrichi avec `component`, `script`, `backend`, `model`, `model_source`, `model_install`.
    - en `pipeline100ml`, l'audit montre explicitement `attribution-gramatical-100ml-xlmr.py` (et non le script legacy 3 fichiers).
  - `pipeline/cli.py`:
    - alias de compatibilite conserve pour l'ancien nom d'etape (`atripusion-gramatical-en-utilisant-les3ficherla` -> `atripusion-gramatical`).
  - `README.md` + `PROJECT_CODE_MAP.md`:
    - correction des references de chemin/fichier: `atripusion-gramatical.py` remplace par les scripts reels (`atripusion-gramatical-en-utilisant-les3ficherla.py` ou `attribution-gramatical-100ml-xlmr.py` selon pipeline).
  - `component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py`:
    - ajout d'un raffinement POS 100ml (`hybrid-rules+context+xlmr-prototypes-v2`) pour ameliorer la precision des etiquettes grammaticales.
    - ajout d'un audit terminal POS explicite: `[grammar-100ml-xlmr][pos] method=... | refined=... | top=...`.
    - publication des metriques POS dans le contexte (`NLP_POS_METHOD`, `NLP_POS_REFINED_COUNT`, `NLP_POS_TOTAL`, `NLP_POS_REFINED_RATE`, `NLP_POS_TOP`).
  - `component/table_extraction/table_extraction_lib.py`:
    - integration des principes de `recherche.txt` pour la detection des tableaux OCR:
      - conservation de la geometrie texte (espaces/indents) au lieu de normaliser trop tot,
      - features par ligne (`token_starts`, `gap_sizes`, `large_gap_count`, ratios alpha/numerique, alignement voisin),
      - score de tabularite par ligne + detection d'en-tete probable,
      - creation d'ancrages de colonnes (positions X stables) depuis l'en-tete,
      - affectation des segments a la colonne la plus proche (nearest anchor),
      - extension/fermeture du bloc tableau selon compatibilite d'ancrage et ruptures.
    - fusion des blocs `anchor-geometry` avec les blocs heuristiques historiques pour robustesse mixte.
    - version moteur passee a `table-<profile>-unified-v3-anchor-geometry`.
  - validation runtime:
    - `python main.py documents/image2tab.webp --pipeline pipeline100ml --upto table-extraction --es-nlp-level off`
      - resultat: `docs=1 | tables=2 | rows=13`.
    - `python main.py documents/signettab.png --pipeline pipeline100ml --upto table-extraction --es-nlp-level off`
      - resultat: `docs=1 | tables=1 | rows=3`.
  - `pipeline/components.py`:
    - le resume de l'etape `atripusion-gramatical` affiche maintenant `pos=...` et `pos_refined=...` quand disponible.
  - `component/fusion_resultats.py`:
    - le bloc `components.attribution_grammaticale` expose maintenant l'audit POS (`pos_method`, `pos_refined_count`, `pos_total`, `pos_refined_rate`, `pos_top_tags`).
- 2026-03-17:
  - `component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py`:
    - nouveau composant d'attribution grammaticale dedie a `pipeline100ml`.
    - backend Transformer XLM-R (`xlm-roberta-base`) multi-langue FR/EN/AR.
    - auto-install du modele si absent localement (tentative Hub/cache), puis fallback automatique.
    - production des sorties standard `NLP_*` (`NLP_ANALYSES`, `NLP_TOKENS`, `NLP_ENTITIES`, `NLP_LANGUAGE_STATS`) pour compatibilite totale avec le reste du pipeline.
    - fallback hash local automatique si le modele reste indisponible.
  - `pipeline/orchestrator.py`:
    - `Pipeline100MLOrchestrator` branche maintenant l'etape grammaire vers `component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py` (sans impacter `default` et `pipeline50ml`).
  - `pipeline/cli.py`:
    - aide `--pipeline` mise a jour pour indiquer la grammaire XLM-R sur `pipeline100ml`.
  - `component/table_extraction/table_extraction_lib.py` (amelioration extraction tableaux dense/partiels):
    - ajout du champ `reference` (code produit) dans `line_items` + `detected_columns`,
    - heuristiques OCR renforcees pour choisir la meilleure variante d'image (autocontrast + sharpen + upscale + fallback threshold),
    - fallback OCR force quand extraction locale reste trop pauvre (lignes uniquement code/sans valeurs),
    - suppression des tableaux faibles/redondants quand un tableau riche est present,
    - normalisation des labels de petits tableaux (totaux, montant a payer, timbre),
    - sortie table enrichie: `table_type`, `shape`, `header_map` et lignes propres pour grands/petits tableaux.
  - validation cible `documents/image2tab.webp`:
    - suppression du premier tableau redondant,
    - extraction table complete: `tables=2`, `rows=13` (10 lignes produits + 3 lignes totaux),
    - colonnes detectees: `reference`, `product`, `quantity`, `unit_price`, `total`.
  - `component/table_extraction/table-extraction.py` + `component/table_extraction/table_extraction_lib.py`:
    - passage a un composant runtime unique pour `pipeline50ml` et `pipeline100ml`.
    - suppression des doublons scripts `table-extraction-50ml.py` et `table-extraction-100ml.py`.
    - heuristiques renforcees pour tableaux OCR denses/serres:
      - split de lignes compactes (single-space) avec detection queue numerique,
      - detection robuste des lignes 2 colonnes (code produit + montant),
      - recuperation de codes produits isoles sous en-tete tableau (`c1009`, `c1010`) meme sans montant OCR,
      - fallback OCR table-first (`pytesseract --psm 3/4/6`, variantes autocontrast/threshold/sharpen/upscale) quand extraction locale trop faible,
      - filtrage du bruit non-table (date, IBAN/SWIFT, tel, entetes administratifs),
      - mapping colonnes ameliore pour tables 4 colonnes (`quantity/product/unit_price/total`),
      - normalisation stricte des montants/quantites pour eviter faux positifs.
    - sorties context conservees: `TABLE_EXTRACTIONS`, `TABLE_EXTRACTIONS_50ML`, `TABLE_EXTRACTIONS_100ML`.
    - chaque document expose: `tables_count`, `rows_total`, `detected_columns`, `totals`, `line_items`, `tables`.
  - `pipeline/components.py` + `pipeline/orchestrator.py` + `pipeline/cli.py`:
    - ajout d'une nouvelle etape `table-extraction`.
    - insertion de l'etape dans `Pipeline50MLOrchestrator` et `Pipeline100MLOrchestrator` entre grammaire et liaison inter-docs.
    - ajout de `table-extraction` dans `--only`, `--upto`, `--start`.
  - `component/fusion_resultats-50ml.py` + `component/fusion_resultats-100ml.py`:
    - integration des resultats tableaux dans `documents[].extraction.table_extraction`.
    - ajout des resumes composant:
      - `documents[].components.table_extraction_50ml`,
      - `documents[].components.table_extraction_100ml`.
    - ajout du compteur pipeline:
      - `pipeline.ml50.tables_docs_count`,
      - `pipeline.ml100.tables_docs_count`.
  - validation technique:
    - `documents/image2tab.webp`:
      - `pipeline50ml`: `docs=1 | tables=2 | rows=13`,
      - `pipeline100ml`: `docs=1 | tables=2 | rows=13`,
      - fallback OCR recupere les lignes produits manquantes et les colonnes numeriques.
    - `documents/signettab.png`:
      - `pipeline50ml`: `docs=1 | tables=1 | rows=3`,
      - `pipeline100ml`: `docs=1 | tables=1 | rows=3`,
      - lignes produits propres conservees (`quantity/product/unit_price/total`), bruit administratif retire.
  - structure:
    - creation du dossier `component/table_extraction/` pour centraliser le moteur commun.
    - composant runtime unique: `component/table_extraction/table-extraction.py`.
    - moteur commun: `component/table_extraction/table_extraction_lib.py`.
  - `component/clasification.py`:
    - ajout d'un audit de score par type documentaire (`scores_audit`) pour rendre chaque score explicable.
    - pour chaque type (ex: `FACTURE`, `FORMULAIRE`), stockage des mots-cles ayant contribue au score avec:
      - `keyword`, `bucket` (strong/medium/weak/negative/anti_confusion), `count`, `score`.
      - vue compacte `matched_keywords_compact` (ex: `INVOICE(x82,+492,strong)`).
    - impression terminal enrichie:
      - nouvelle ligne `score_audit:` apres la ligne `[classification] ... scores: {...}`.
      - permet de voir directement quels mots ont produit les scores non nuls.
    - `RESULTS[].scores_audit` est maintenant exporte dans la sortie fusion (`fusion_output.json`) via le bloc `documents[].classification`.
  - `component/extraction-regles-yaml.py`:
    - nettoyage strict des valeurs techniques pour les pipelines `pipeline50ml` et `pipeline100ml` (utilisees via `extraction-regles-50ml.py` et `extraction-regles-100ml.py`).
    - ajout d'une normalisation robuste des caracteres parasites (`\\n`, `\\r`, `\\t`, `\\xa0`, controle Unicode) avant extraction.
    - ajout de normaliseurs typés:
      - `email`: conserve uniquement l'adresse email valide (sans quotes/retours ligne).
      - `phone`: conserve uniquement le numero (digits purs ou `+digits`, sans libelles type `(Mobile ...)`).
      - `url`: conserve uniquement l'URL brute.
      - `date`: extraction de la date pure (formats numeriques + formats texte FR/EN, ex: `Lundi 12 mars 2024`) sans reste de phrase.
    - `detecteurs` et `normalize_value_by_type` alignes pour garantir des valeurs propres et dedupees dans `EXTRACTIONS.fields.*.matches[].value`.
  - validation locale:
    - test direct moteur YAML sur exemples utilisateur:
      - `jean-pierre.durand@service-public.fr\\n'` -> `jean-pierre.durand@service-public.fr`
      - `0550123456\\xa0(Mobile Ooredoo)\\n` -> `0550123456`
      - `Date de signature: Lundi 12 mars 2024 ...` -> `Lundi 12 mars 2024`
- 2026-03-16:
  - `component/liaison-inter-docs.py`:
    - nouveau composant de liaison inter-documents base topics (`ML50_TOPICS`/`ML100_TOPICS`) + matching lexical phrase-a-phrase.
    - produit un audit explicite des correspondances: `shared_terms`, `shared_topics`, `phrase_a`/`phrase_b` (`page_index`, `sent_index`, `text_excerpt`), `score`.
    - publie dans le contexte: `INTERDOC_ANALYSIS`, `INTERDOC_LINKS`, `INTERDOC_DOC_LINKS`.
    - quality boost `shared_terms`: filtrage semantique via POS/lemma (`NLP_TOKENS`), stopwords/noise et bonus des signaux `topics + classification`.
    - enrichissement audit: `phrase_a`/`phrase_b` portent maintenant `document_id` + `filename`, et `shared_topics[]` inclut `doc_a_examples` / `doc_b_examples` (extraits de phrases par document).
  - `pipeline/components.py` + `pipeline/orchestrator.py` + `pipeline/cli.py`:
    - ajout de l'etape `liaison-inter-docs` dans les 3 pipelines (`default`, `pipeline50ml`, `pipeline100ml`) entre grammaire et Elasticsearch.
    - ajout de `liaison-inter-docs` dans les options `--only`, `--upto`, `--start`.
  - `component/fusion_resultats.py`:
    - ajout du bloc top-level `cross_document_analysis` (liens + audit complet, sans duplication par document).
    - ajout du bloc `document.cross_document` (`linked_documents_count`, `link_ids`) pour referencer les liens concernes.
    - ajout du resume composant `components.liaison_inter_docs`.
    - correction fusion locale multi-doc: construit des payloads pour chaque document (`TOK_DOCS`/`FINAL_DOCS`) au lieu de ne garder que le premier.
    - effet: les enrichissements ML (`document_primary_topics`/`document_topics`) sont maintenant presents pour chaque document dans `fusion_output.json`.
  - `pipeline/orchestrator.py`:
    - pipelines `Pipeline50MLOrchestrator` et `Pipeline100MLOrchestrator` incluent explicitement l'etape grammaire (`atripusion-gramatical`) dans la sequence runtime.
  - `pipeline/cli.py`:
    - help `--pipeline` corrigee pour reflecter l'usage de la grammaire dans `pipeline50ml` et `pipeline100ml`.
  - `component/fusion_resultats-50ml.py`:
    - ajout d'un filtre grammatical des topics bases sur `NLP_TOKENS` (POS + token/lemma) pour retirer pronoms, determinants, conjonctions, auxiliaires, adverbes et ponctuation.
    - filtrage applique aux niveaux document et chunk (`document_topics`, `document_primary_topics`, `chunk_topics`, `chunk_primary_topic`).
    - ajout du compteur `components.tokenisation_layout_50ml.topics_removed_by_grammar`.
  - `component/fusion_resultats-100ml.py`:
    - meme filtrage grammatical des topics que la version 50ml (document + chunk).
    - ajout du compteur `components.tokenisation_layout_100ml.topics_removed_by_grammar`.
  - `component/tokenisation-layout-50ml.py` + `component/tokenisation-layout-100ml.py`:
    - enrichment de `_STOPWORDS` (pronoms/mots-outils FR/EN + `plus`, `qui`, `que`, etc.) pour eviter des topics grammaticaux des l'etape `[topic-doc]`.
  - validation runtime:
    - `python main.py documents/testword.docx --pipeline pipeline50ml --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens`
    - `python main.py documents/testword.docx --pipeline pipeline100ml --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens`
    - `python main.py documents/contrat_regex_test_corpus_fr_en_ar.pdf documents/contras-14page.pdf --pipeline pipeline50ml --es-nlp-level off`
    - `python main.py documents/testwordaudi.docx documents/testwordlambo.docx documents/testwordvw.docx --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens`
    - resultat observe: suppression des topics grammaticaux (ex: `plus`) des sorties `[topic-doc]`, `[ml50-topic]`, `[ml100-topic]` et `fusion_output.json`.
    - resultat observe liaison inter-docs: `links=1` avec audit phrases (`contras-14page.pdf <-> contrat_regex_test_corpus_fr_en_ar.pdf`).
    - resultat observe fusion multi-doc: `docs=3` en sortie fusion et `document_primary_topics` visibles pour `testwordaudi.docx`, `testwordlambo.docx`, `testwordvw.docx`.
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
  - `pipeline/cli.py`:
    - ajout de la selection pipeline `--pipeline default|pipelinorchestrator|pipeline50ml|pipeline100ml`,
    - ajout du pilotage par variable d'environnement `PIPELINE_DEFAULT` (ex: `pipelinorchestrator`, `pipeline50ml` ou `pipeline100ml`; fallback `PIPELINE_PROFILE`),
    - ajout du pilotage direct dans le code via `PIPELINE_DEFAULT_CODE` (utilise si aucune variable n'est fournie),
    - log explicite de la pipeline selectionnee,
    - ajout des options de contexte `ML100_MODEL_NAME`, `ML100_MAX_LENGTH`, `ML100_BATCH_SIZE`, `ML100_HASH_FALLBACK_DIM`.
  - `pipeline/orchestrator.py`:
    - nouvel orchestrateur `Pipeline50MLOrchestrator`,
    - pipeline alternative sans composant grammaire,
    - etapes: pretraitement -> ocr routing -> output-txt -> classification -> tokenisation 50ml -> elasticsearch -> extraction 50ml -> fusion 50ml.
  - `pipeline/orchestrator.py`:
    - nouvel orchestrateur `Pipeline100MLOrchestrator`,
    - pipeline alternative sans composant grammaire,
    - etapes: pretraitement -> ocr routing -> output-txt -> classification -> tokenisation 100ml -> elasticsearch -> extraction 100ml -> fusion 100ml.
  - `pipeline/orchestrator.py` + `pipeline/cli.py`:
    - nouvel ordre pipeline: `clasification` executee juste apres `output-txt`, avant `tokenisation-layout`.
    - expose desormais `PIPELINE_PROFILE` et `PIPELINE_STEPS` dans le contexte.
  - `component/tokenisation-layout-50ml.py`:
    - execute la tokenisation/layout standard puis enrichit avec embeddings FastText-like (subword hashing),
    - produit des vecteurs `mot/chunk/document` (`ML50_*`),
    - ajoute un topic extractor (`ML50_TOPICS`) + topics par chunk (`ML50_CHUNK_VECTORS.chunk_primary_topic/chunk_topics`) + `document_primary_topics` document (top 2),
    - topic extractor renforce: normalisation linguistique, filtrage bruit OCR, n-grams, scoring TF-IDF pondere, boost via `RESULTS.keyword_matches`,
    - ajoute un affichage terminal explicite par document: `[topic-doc] ... document_primary_topics/document_top_topics`.
    - genere `NLP_ANALYSES`/`NLP_TOKENS` minimaux pour la sync ES sans composant grammaire.
  - `component/extraction-regles-50ml.py`:
    - execute extraction YAML (sans regex de champs) via `extraction-regles-yaml.py` puis ajoute un scoring BM25 par chunk,
    - enrichit `EXTRACTIONS[].bm25` + publie `BM25_RESULTS`.
  - `component/fusion_resultats-50ml.py`:
    - execute la fusion standard puis enrichit `fusion_output.json` avec:
      - `document.ml50` (vectors/topics),
      - `document.ml50.document_primary_topics` (2 topics principaux),
      - `document.ml50.document_topics` toujours renseigne (fallback depuis chunks si besoin),
      - `extraction.bm25` (score retrieval),
      - `components.tokenisation_layout_50ml` et `components.extraction_regles_50ml`,
      - `pipeline.profile=\"pipeline50ml\"` + `pipeline.ml50`.
    - ajoute un print terminal par document `[ml50-topic]` pour rendre visibles les topics dans `outputgeneralterminal.txt`.
  - `component/tokenisation-layout-100ml.py`:
    - execute la tokenisation/layout standard puis calcule des embeddings Transformer (BERT/XLM-R) avec mean pooling,
    - produit 1 embedding par chunk + 1 embedding document (`ML100_*`),
    - fallback hash local automatique si modele indisponible,
    - topic extractor renforce: normalisation linguistique, filtrage bruit OCR, n-grams, scoring TF-IDF pondere, boost via `RESULTS.keyword_matches`,
    - ajoute un topic extractor (`ML100_TOPICS`) + `document_primary_topics`/`document_topics`.
  - `component/extraction-regles-100ml.py`:
    - execute extraction YAML (sans regex de champs) via `extraction-regles-yaml.py` puis ajoute un scoring BM25 par chunk (tag log 100ml).
  - `component/extraction-regles-yaml.py`:
    - nouveau moteur d'extraction par YAML (labels + detecteurs date/email/telephone/url/amount/currency),
    - selection des champs selon `classification.doc_type`,
    - sortie compatible `EXTRACTIONS` (rule_id/type/matches) pour fusion + BM25.
  - `config/ruleset_routes.yaml` + `rules/*.yaml`:
    - ajout du routage et des regles metier YAML (`common.yaml`, `CONTRAT.yaml`, `FACTURE.yaml`, `BON_DE_COMMANDE.yaml`) utilises par les pipelines 50ml/100ml.
  - `component/fusion_resultats-100ml.py`:
    - execute la fusion standard puis enrichit `fusion_output.json` avec:
      - `document.ml100` (vectors/topics),
      - `components.tokenisation_layout_100ml` et `components.extraction_regles_100ml`,
      - `pipeline.profile=\"pipeline100ml\"` + `pipeline.ml100`.
    - ajoute un print terminal par document `[ml100-topic]`.
  - `pipeline/components.py`:
    - `ClassificationComponent`: sync ES differee tant que les docs ne sont pas indexes (`ES_DOC_IDS` absents) pour eviter une sync prematuree.
    - `ElasticsearchComponent`: reporting enrichi avec `classification_input` et `classification_synced`.
  - `component/elasticsearch.py`:
    - ajoute la sync classification (`RESULTS`) directement dans l'etape Elasticsearch apres indexation des documents,
    - expose `ES_CLASSIFICATION_SYNCED` dans le contexte et le log composant.
  - `component/fusion_resultats.py`:
    - corrige `file.page_count` (base sur les vraies pages document et plus sur des wrappers doc),
    - corrige `document_structure.pages` (retourne des pages reelles avec `page_index`, plus des objets document),
    - dedup des `regex_extractions` sur le document courant pour supprimer les doublons.
  - `component/extraction-regles.py`:
    - dedup des documents d'entree (meme doc_id/filename) avec priorite a la version `text` vs `image_only`,
    - evite les sorties `EXTRACTIONS` dupliquees sur un meme fichier.
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
  - noeuds execution (CLI -> orchestrateur -> 10 composants)
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
