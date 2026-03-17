# Project Code Map (DMS Core)

Date d'audit: 2026-03-16

## 1) Scope de l'audit
- Depot analyse: `/home/mourad/Bureau/DMS/core`
- Python files analyses: 27
- Fonctions/classes indexees: 528 (voir `FUNCTION_INDEX.txt`)
- Regles metier JSON/YAML: `rules/*.json` + `rules/*.yaml` + `classification/*.json` + `config/ruleset_routes.json` + `config/ruleset_routes.yaml`

## 2) Points d'entree
- CLI principal: `main.py` -> `pipeline.cli:main`
- CLI package: `orchestre` (defini dans `pyproject.toml`)
- Parsing des options: `pipeline/cli.py`
- Orchestrateurs:
  - `pipeline/orchestrator.py` (contient `PipelineOrchestrator` + `Pipeline50MLOrchestrator` + `Pipeline100MLOrchestrator`)
- Wrappers d'execution des composants: `pipeline/components.py`

## 3) Pipeline reel (ordre d'execution)
- Pipeline `default`:
  1. `pretraitement-de-docs`
  2. `si-image-pretraiter-sinonpass-le-doc`
  3. `output-txt`
  4. `clasification`
  5. `tokenisation-layout`
  6. `atripusion-gramatical-en-utilisant-les3ficherla`
  7. `liaison-inter-docs`
  8. `elasticsearch`
  9. `extraction-regles`
  10. `fusion-resultats` (debug/fusion finale, non bloquant en erreur)
- Pipeline `pipeline50ml`:
  1. `pretraitement-de-docs`
  2. `si-image-pretraiter-sinonpass-le-doc`
  3. `output-txt`
  4. `clasification`
  5. `tokenisation-layout` (`component/tokenisation-layout-50ml.py`):
     - tokenisation/layout standard + enrichissement FastText-like (subword hashing),
     - vecteurs mot/chunk/document,
     - topic extraction par chunk (`chunk_primary_topic`, `chunk_topics`) + topics document,
     - generation `NLP_*` minimale (provisoire, puis remplacee par la sortie grammaire).
  6. `atripusion-gramatical-en-utilisant-les3ficherla`
  7. `liaison-inter-docs` (`component/liaison-inter-docs.py`):
     - detection de liens inter-documents par overlap de topics + matching phrase-a-phrase auditable.
  8. `elasticsearch`
  9. `extraction-regles` (`component/extraction-regles-50ml.py`):
     - extraction YAML (sans regex de champs) pilotee par classification/doc_type + scoring BM25 par chunk.
  10. `fusion-resultats` (`component/fusion_resultats-50ml.py`):
     - fusion standard + ajout `ml50` + BM25 dans `fusion_output.json`,
     - filtrage des topics grammaticaux (pronoms/determinants/conjonctions/adverbes) via `NLP_TOKENS` du composant grammaire.
- Pipeline `pipeline100ml`:
  1. `pretraitement-de-docs`
  2. `si-image-pretraiter-sinonpass-le-doc`
  3. `output-txt`
  4. `clasification`
  5. `tokenisation-layout` (`component/tokenisation-layout-100ml.py`):
     - tokenisation/layout standard + embeddings Transformer (BERT/XLM-R) avec mean pooling,
     - 1 embedding par chunk + 1 embedding document,
     - topic extraction par chunk (`chunk_primary_topic`, `chunk_topics`) + topics document,
     - generation `NLP_*` minimale (provisoire, puis remplacee par la sortie grammaire),
     - fallback hash local si modele indisponible.
  6. `atripusion-gramatical-en-utilisant-les3ficherla`
  7. `liaison-inter-docs` (`component/liaison-inter-docs.py`):
     - detection de liens inter-documents par overlap de topics + matching phrase-a-phrase auditable.
  8. `elasticsearch`
  9. `extraction-regles` (`component/extraction-regles-100ml.py`):
     - extraction YAML (sans regex de champs) pilotee par classification/doc_type + scoring BM25 par chunk.
  10. `fusion-resultats` (`component/fusion_resultats-100ml.py`):
     - fusion standard + ajout `ml100` + BM25 dans `fusion_output.json`,
     - filtrage des topics grammaticaux (pronoms/determinants/conjonctions/adverbes) via `NLP_TOKENS` du composant grammaire.

Selection runtime:
- CLI: `--pipeline default|pipelinorchestrator|pipeline50ml|pipeline100ml`
- variable d'environnement par defaut: `PIPELINE_DEFAULT` (ex: `pipelinorchestrator`, `pipeline50ml` ou `pipeline100ml`; fallback `PIPELINE_PROFILE`)

Reference implementation:
- `pipeline/orchestrator.py`

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

### 4.6 Liaison inter-documents
- composant: `component/liaison-inter-docs.py`
- sortie contexte:
  - `INTERDOC_ANALYSIS` (methode, statistiques, liens)
  - `INTERDOC_LINKS` (liens inter-documents)
  - `INTERDOC_DOC_LINKS` (index doc -> link_ids)
- audit phrase-a-phrase:
  - chaque lien contient `audit.matches[]` avec `phrase_a` / `phrase_b` (page/sentence/text_excerpt), `shared_terms`, `shared_topics`, `score`.
  - `shared_terms` est nettoye pour garder des termes informatifs (stopwords/pronoms/mots vides exclus; priorite aux termes semantiques POS/lemma + topics + signaux classification).

### 4.7 Elasticsearch (optionnel)
- activation: `USE_ELASTICSEARCH`
- conf: `ES_URL`, `ES_INDEX`
- auth HTTP: `ES_USER`, `ES_PASSWORD`, `ES_API_KEY`
- conf NLP: `ES_NLP_LEVEL` (`off|summary|full`), `ES_NLP_INDEX`, `ES_NLP_MAX_FULL_TOKENS`
- auto-start: `ES_AUTO_START`, `ES_START_COMMAND`, `ES_START_COMMANDS`, `ES_START_COMMAND_POSIX`, `ES_START_COMMAND_WINDOWS`, `ES_START_PASSWORD`, `ES_AUTO_START_WAIT_SECONDS`, `ES_AUTO_START_LAUNCH_TIMEOUT`
  - comportement par defaut: auto-start Windows uniquement, Linux/macOS en demarrage manuel.
- sorties: `ES_AVAILABLE`, `ES_DOC_IDS`, `ES_CLASSIFICATION_DOCS`, `ES_EXTRACTION_DOCS`, `ES_AUTO_STARTED`, `ES_AUTO_START_CMD`
- sorties NLP ES: `ES_NLP_SYNC`, `ES_NLP_DOCS_SYNCED`, `ES_NLP_TOKENS_SYNCED`, `ES_NLP_TOKEN_ERRORS`, `ES_NLP_LEVEL_EFFECTIVE`, `ES_NLP_INDEX_EFFECTIVE`

### 4.8 Classification
- sortie: `RESULTS` (doc_type, status, scores, `classification_log`, `keyword_matches`, `anti_confusion_targets`)
- sync ES: `ES_CLASSIFICATION_SYNCED`

### 4.9 Extraction metier
- sortie: `EXTRACTIONS`
- sync ES: `ES_EXTRACTION_SYNCED`
- en `pipeline50ml`: extraction basee YAML (`rules/*.yaml`, `config/ruleset_routes.yaml`) + `EXTRACTIONS[].bm25` + `BM25_RESULTS`
- en `pipeline100ml`: extraction basee YAML (`rules/*.yaml`, `config/ruleset_routes.yaml`) + `EXTRACTIONS[].bm25` + `BM25_RESULTS`

### 4.10 Fusion finale
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
- `component/tokenisation-layout.py`
  - `layout_items`
  - `_transpose_or_group_multicol`
  - `_collect_table_block`
  - `chunk_layout_universal`
  - `chunk_is_noise`
- variante ML50 (embedding/topic/doc-vector):
  - `component/tokenisation-layout-50ml.py`
- variante ML100 (Transformer BERT/XLM-R + pooling):
  - `component/tokenisation-layout-100ml.py`

### 5.6 Changer la liaison inter-documents (topics + audit phrases)
- `component/liaison-inter-docs.py`
  - calcul des liens entre documents,
  - regles de score/threshold,
  - structure de l'audit `phrase_a` / `phrase_b` et champs exposes dans la fusion.

### 5.7 Changer classification documentaire (scores, threshold, priorites)
- code: `component/clasification.py`
- config: `classification/common.json`, `classification/*.json`

### 5.8 Changer extraction metier
- moteur regex (pipeline default): `component/extraction-regles.py`
- moteur YAML (pipelines 50ml/100ml): `component/extraction-regles-yaml.py`
- routage rulesets regex: `config/ruleset_routes.json`
- routage rulesets YAML: `config/ruleset_routes.yaml`
- patterns metier regex: `rules/*.json`
- patterns metier YAML: `rules/*.yaml`
- variante ML50 avec BM25:
  - `component/extraction-regles-50ml.py`
- variante ML100 avec BM25:
  - `component/extraction-regles-100ml.py`

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
- en mode `NLP full`, la fusion charge aussi les tokens NLP depuis ES (`dms_nlp_tokens`) pour le document courant et les structure par page/sentence.
- schema fusion lisible humain + exploitable code:
  - top-level: `schema_version`, `generated_at`, `source`, `documents_count`, `documents`
  - par document: `components` detaille tous les composants du pipeline
- classification detaillee exposee (`scores`, `classification_log`, `keyword_matches`, `anti_confusion_targets`) + `file.size`
- deduplication active: `components` reste volontairement compact (resume) et les details complets restent au niveau document (`classification`, `extraction`, `nlp`, `file`)
- variante ML50:
  - `component/fusion_resultats-50ml.py`
  - enrichit la fusion avec `ml50` (vectors/topics) + BM25
- variante ML100:
  - `component/fusion_resultats-100ml.py`
  - enrichit la fusion avec `ml100` (vectors/topics) + BM25

### 5.11 Changer logique linguistique EN/FR/AR
- orchestrateur langues: `component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py`
- anglais: `component/atrribution-gramatical/engcode.py`
- francais: `component/atrribution-gramatical/frcode.py`
- arabe: `component/atrribution-gramatical/arabcode.py`

## 6) Cartographie des fichiers Python (roles)

### 6.1 Orchestration (`pipeline/`)
- `pipeline/settings.py`: logging, normalize input, context managers cwd/argv, chargement `.env` optionnel.
- `pipeline/components.py`: wrappers des composants scripts + resumes + sync ES.
- `pipeline/orchestrator.py`: contient les 3 orchestrateurs (`PipelineOrchestrator`, `Pipeline50MLOrchestrator`, `Pipeline100MLOrchestrator`) + selection `only/upto/start`.
- `pipeline/cli.py`: CLI + chargement `.env` + tee print vers `outputgeneralterminal.txt`.
- `pipeline/elasticsearch.py`: store HTTP ES + auth + flatten/index + auto-start local ES (POSIX/Windows) + fallback docs + sync classification/extraction + sync NLP (summary/full).

### 6.2 Composants metier (`component/`)
- `pretraitement-de-docs.py`: detect format, determine `text` vs `image_only`, extrait `size`.
- `si-image-pretraiter-sinonpass-le-doc.py`: split OCR/native + preprocess image.
- `output-txt.py`: OCR tesseract + extraction native multi-format + `FINAL_DOCS` (+ `size`).
- `tokenisation-layout.py`: language detect + sentence/layout chunking + table/multicol + `TOK_DOCS` (+ `size`).
- `tokenisation-layout-50ml.py`: tokenisation/layout + embeddings FastText-like + topics + vectors doc/chunk/word + `NLP_*` minimal (provisoire avant grammaire).
  - topics chunk-level (`chunk_primary_topic`, `chunk_topics`) + top-2 document (`document_primary_topics`).
  - scoring topics ameliore (n-grams, dedupe semantique, boost keywords classification, anti-bruit OCR).
- `tokenisation-layout-100ml.py`: tokenisation/layout + embeddings Transformer (BERT/XLM-R) + pooling mean + vectors chunk/doc + topics + `NLP_*` minimal (provisoire avant grammaire).
  - fallback hash si modele non disponible localement.
  - scoring topics ameliore (n-grams, dedupe semantique, boost keywords classification, anti-bruit OCR).
- `atrribution-gramatical/*.py`: POS/lemma/NER per language + notebook style runners.
- `liaison-inter-docs.py`: lie les documents entre eux via topics + recouvrement lexical phrase-a-phrase, puis publie un audit des matches.
  - filtre qualite `shared_terms`: supprime mots non-significatifs (`son`, `tout`, etc.) et favorise termes metier/juridiques.
- `elasticsearch.py`: step script pour index/fetch docs ES.
- `clasification.py`: keyword scoring classification + details matches (`classification_log`, `keyword_matches`).
- `extraction-regles.py`: regex extractors selon doc_type.
- `extraction-regles-yaml.py`: extraction sans regex de champs (labels/detecteurs) selon doc_type via YAML.
- `extraction-regles-50ml.py`: extraction-regles-yaml + scoring BM25 sur chunks.
- `extraction-regles-100ml.py`: extraction-regles-yaml + scoring BM25 sur chunks.
- `fusion_resultats.py`: build JSON fusion structure par document (`documents[]`) depuis context + ES.
  - ajoute `cross_document_analysis` (liens + audit) et `document.cross_document` (references link_ids).
- `fusion_resultats-50ml.py`: fusion_resultats + enrichissement ML50/BM25.
  - force la visibilite `ml50.document_topics`/`ml50.document_primary_topics` dans `fusion_output.json` (fallback depuis les topics de chunks si `ML50_TOPICS` absent/mal aligne).
  - applique un filtrage grammatical des topics via `NLP_TOKENS` (POS/lemma) pour retirer pronoms/mots-outils.
  - affiche un resume terminal par document: `[ml50-topic] <filename> | document_primary_topics=[...] | document_top_topics=[...]`.
- `fusion_resultats-100ml.py`: fusion_resultats + enrichissement ML100/BM25.
  - produit `document.ml100` + `components.tokenisation_layout_100ml` + `components.extraction_regles_100ml`.
  - applique un filtrage grammatical des topics via `NLP_TOKENS` (POS/lemma) pour retirer pronoms/mots-outils.

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
- `component/extraction-regles.py` compile les regex sans garde plus fine que `re.error`; en cas de pattern invalide, le champ est skippe (comportement tolerant).
- `component/fusion_resultats.py` est clairement oriente "debug/fusion", pas schema strict valide via validation formelle.
- La couche `pipeline/` est proprement separee et sert de facade stable autour des scripts notebooks.

## 10) Index complet des fonctions/classes
- Voir `FUNCTION_INDEX.txt` pour la liste exhaustive `file:line:def/class`.
- Ce fichier est la reference la plus rapide pour localiser une modification precise.

## 11) Changelog code
- 2026-03-17:
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
    - pipelines `Pipeline50MLOrchestrator` et `Pipeline100MLOrchestrator` incluent explicitement l'etape grammaire (`atripusion-gramatical-en-utilisant-les3ficherla`) dans la sequence runtime.
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
