# DMS Pipeline Orchestrator

Ce dépôt regroupe des scripts de traitement documentaire (prétraitement, OCR, tokenisation, grammaire, classification) et un orchestrateur léger qui les enchaîne **sans modifier leur logique métier**.

## Architecture
- `pretraitement-de-docs.py` → `si-image-pretraiter-sinonpass-le-doc` → `output-txt.py` → `clasification.py` → `tokenisation-layout` → `atripusion-gramatical` → `liaison-inter-docs.py` → `elasticsearch.py` → `extraction-regles.py` → `fusion_resultats.py`
- `component/tokenisation_layout/` : scripts de tokenisation/layout (`default`, `50ml`, `100ml`)
- `component/extraction/` : scripts d'extraction (`regex`, `yaml`, `50ml`, `100ml`)
- `component/fusion_resultats.py` : fichier unique de fusion pour `default`, `pipeline50ml`, `pipeline100ml`
- `pipeline/` : couche d'orchestration open-source friendly  
  - `settings.py` : logging, helpers (argv isolation, cwd, normalisation des entrées)  
  - `components.py` : wrappers `Component` pour chaque script  
  - `orchestrator.py` : assemble l'ordre des composants  
  - `cli.py` : parsing CLI et point d'entrée
- `main.py` : shim pour lancer le CLI (`python main.py ...` ou `orchestre ...` via console_script).

## Exécution
```bash
python main.py documents/englais.docx
# ou
python -m pipeline.cli documents/englais.docx
```

## Exécution avec Elasticsearch
Le pipeline peut maintenant indexer les documents tokenisés dans Elasticsearch à l'étape
`elasticsearch`, puis:
- relire le texte/passages/mots depuis Elasticsearch pour `extraction-regles` (et `clasification` si documents déjà indexés)
- écrire les résultats de classification, d'extraction et de NLP dans Elasticsearch
- construire `fusion_output.json` depuis Elasticsearch (mode debug/inspection)
  - `fusion_resultats.py` est optionnel (debug): s'il est absent ou en erreur, le pipeline principal continue.

```bash
python main.py documents/contrat_regex_test_corpus_fr_en_ar.pdf \
  --use-elasticsearch \
  --es-url http://localhost:9200 \
  --es-index dms_documents
```

## Téléchargements automatiques (global)
Ce dépôt peut télécharger automatiquement des ressources au premier lancement, selon les composants exécutés.

### 1) Pipeline100 grammaire XLM-R
Composant:
- `component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py`

Modèle par défaut:
- `xlm-roberta-base`

Artefacts téléchargés automatiquement (si absents):
- `config.json`
- `tokenizer_config.json`
- `tokenizer.json`
- `special_tokens_map.json`
- `sentencepiece.bpe.model`
- poids du modèle (`model.safetensors` ou `pytorch_model.bin`)

Emplacements:
1. `ML100_MODEL_LOCAL_DIR` si défini (utilise ce dossier, pas de téléchargement)
2. sinon cache projet: `/home/mourad/Bureau/DMS/core/component/atrribution-gramatical/.hf_model_cache`
3. fallback `transformers` direct Hub: `~/.cache/huggingface/hub` (ou `HF_HOME` / `TRANSFORMERS_CACHE`)

Variables:
- `ML100_MODEL_NAME`
- `ML100_MODEL_LOCAL_DIR`
- `ML100_MODEL_CACHE_DIR`
- offline: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `LANG_PIPE_OFFLINE=1`

Audit terminal:
- `[grammar-100ml-xlmr] ... source=...`
- `[grammar-100ml-xlmr][model] auto-installed:... | remote-hub:... | local-dir:...`

### 2) Pipeline100 tokenisation embeddings Transformer
Composant:
- `component/tokenisation_layout/tokenisation-layout-100ml.py`

Téléchargement automatique possible:
- modèle `ML100_MODEL_NAME` (par défaut `xlm-roberta-base`) via `transformers` (`AutoTokenizer.from_pretrained`, `AutoModel.from_pretrained`)

Emplacement:
- cache Hugging Face global (`~/.cache/huggingface/hub`, ou `HF_HOME` / `TRANSFORMERS_CACHE`)

### 3) Grammaire EN/FR (pipeline default et pipeline50ml)
Composants:
- `component/atrribution-gramatical/engcode.py`
- `component/atrribution-gramatical/frcode.py`

Téléchargements automatiques possibles:
- NLTK (EN): `punkt`, `averaged_perceptron_tagger`, `wordnet`, `omw-1.4`
- Modèle NER EN: `dslim/bert-base-NER`
- Modèle NER FR: `Davlan/bert-base-multilingual-cased-ner-hrl`

Emplacements:
- NLTK data: `~/nltk_data` (ou variable `NLTK_DATA`)
- modèles HF: `~/.cache/huggingface/hub` (ou `HF_HOME` / `TRANSFORMERS_CACHE`)

### 4) Tokenisation layout classique
Composant:
- `component/tokenisation_layout/tokenisation-layout.py`

Téléchargement automatique possible:
- NLTK: `punkt`, `punkt_tab`

Emplacement:
- `~/nltk_data` (ou `NLTK_DATA`)

### 5) Dépendances non téléchargées automatiquement
- `tesseract` (OCR): requis système, non installé automatiquement par le code
- `camel_tools` + données arabes (`morphology-db-msa-r13`, `ner-arabert`): le code donne les commandes, mais n’installe pas automatiquement

## Maintenance / Open Source
- Code orchestrateur typé et découpé par responsabilités (helpers vs composants vs CLI).
- Pas de dépendance aux chemins Windows dans le code d'orchestration (chemins relatifs au repo).
- Journalisation dans `orchestre.log`.
- Paquet installable : `pip install -e .` puis `orchestre ...`.

## Règles respectées
- Aucun algorithme interne des scripts métiers n'est modifié ni copié.
- Orchestration uniquement : passage des sorties en entrées, validations, logs lisibles.









ce que Elasticsearch stocke dans son index dms_documents :

  {
    "_id": "af6633a1-82f8-405e-a175-b79666201615",
    "_source": {
      "doc_id": "af6633a1-82f8-405e-a175-b79666201615",
      "filename": "contrat_regex_test_corpus_fr_en_ar.pdf",
      "content": "text",
      "extraction": "native:pdf:pypdf",
      "paths": ["/home/mourad/Bureau/DMS/core/documents/contrat_regex_test_corpus_fr_en_ar.pdf"],
      "page_count_total": 12,

      "pages": [
        {
          "page_index": 1,
          "lang": "fr",
          "source_path": ".../contrat_regex_test_corpus_fr_en_ar.pdf",
          "text": "Corpus de test - Contrats / Contracts / عقود ..."
        }
      ],
      "passages": [
        {
          "page_index": 1,
          "layout_kind": "multicol_col",
          "start": 0,
          "end": 600,
          "text": "Corpus de test - Contrats / Contracts / عقود ..."
        }
      ],
      "words": ["corpus", "contrat", "agreement", "..."],
      "full_text": "Texte complet du document ...",
      "detected_languages": ["fr", "en", "ar"],

      "doc_type": "CONTRAT",
      "classification_status": "OK",
      "classification_updated_at": "2026-03-04T10:32:12.711319+00:00",
      "classification": {
        "doc_type": "CONTRAT",
        "status": "OK",
        "scores": {
          "CONTRAT": 202,
          "FACTURE": -179,
          "BON_DE_COMMANDE": -202
        }
      },

      "rule_extraction": {
        "doc_id": "...",
        "doc_type": "CONTRAT",
        "classification_status": "OK",
        "fields_count": 11,
        "fields_with_matches": ["doc_date", "titre_contrat", "date_signature", "..."]
      },
      "rule_extraction_payload": "{... JSON complet de l'extraction ...}",
      "rules_fields_count": 11,
      "rules_fields_matched": ["doc_date", "titre_contrat", "..."],
      "extraction_updated_at": "2026-03-04T10:46:16.948506+00:00"
    }
  }

ES garde:

  - le texte complet + pages + passages + mots,
  - la classification,
  - l’extraction de règles (résumé + payload complet),
  - les métadonnées (fichier, langue, timestamps, type document).
