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
# ou
./main.py documents/englais.docx
# ou
./run-dms documents/englais.docx
```

Important:
- ne lance jamais le document lui-meme comme commande shell
- faux:
```bash
/home/mourad/Bureau/DMS/core/documents/testwordvw.docx --use-elasticsearch
```
- correct:
```bash
./run-dms /home/mourad/Bureau/DMS/core/documents/testwordvw.docx --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens
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

## `index.html` -> Backend API (detail complet)
Le front [index.html](/home/mourad/Bureau/DMS/core/index.html) n'execute pas le pipeline directement.
Il envoie les fichiers au backend local [local_api.py](/home/mourad/Bureau/DMS/core/local_api.py), qui lance ensuite `main.py`.

### 1) Lancer le backend API
```bash
python local_api.py --host 0.0.0.0 --port 8765
```

Le terminal affiche les URLs de service, par exemple:
- `http://127.0.0.1:8765` (meme machine)
- `http://IP_DE_TA_MACHINE:8765` (autre machine du reseau local)

### 2) Adresse API utilisee par `index.html`
Dans `index.html`, `API_BASE` est calcule ainsi:
- si la page est servie depuis `:8765`, le front utilise `window.location.origin`
- sinon fallback explicite: `http://127.0.0.1:8765`

Consequence:
- front et backend sur la meme machine: `127.0.0.1:8765` fonctionne
- front sur une autre machine: il faut appeler `http://IP_DU_BACKEND:8765` (pas `127.0.0.1`)

### 3) Endpoints exposes par le backend
- `GET /`
  - sert la page `index.html`
- `POST /api/run`
  - recoit les fichiers uploades, les sauve en temporaire, puis lance le pipeline
- `GET /api/status`
  - retourne l'etat du job courant (idle/running/completed/failed) + progression
- `OPTIONS /api/run` et `OPTIONS /api/status`
  - preflight CORS

Implementation backend: [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py)

### 4) Format exact de la requete `POST /api/run`
Content-Type requis:
- `multipart/form-data`

Champs fichier acceptes:
- `files` (recommande)
- `files[]`
- `file`

Si aucun fichier n'est recu:
- reponse `400 Bad Request`

Si un job tourne deja:
- reponse `409 Conflict`

Reponse normale:
- `202 Accepted`
- JSON avec `ok=true`, `job_id`, commande lancee et metadonnees job

### 5) Commande reelle lancee par le backend
Le backend construit et execute:
```bash
python main.py <fichiers_uploades_temp> --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens
```

Les fichiers selectionnes dans le navigateur sont copies dans un dossier temporaire (`/tmp/...`) avant execution.

### 6) Suivi temps reel dans la page
`index.html` interroge periodiquement `GET /api/status` pour afficher:
- etape courante (`current_step`)
- pourcentage (`progress_percent`)
- derniere ligne utile (`last_log_line`)

Quand `status=completed`:
- la page affiche "Traitement termine"

Quand `status=failed`:
- la page affiche le `returncode` et la derniere ligne de log

### 7) Exemple cURL
```bash
curl -X POST \
  -F "files=@documents/signettab.png" \
  http://127.0.0.1:8765/api/run
```

Puis:
```bash
curl -s http://127.0.0.1:8765/api/status
```

### 8) CORS
Le backend renvoie:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, POST, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type, Authorization`

Donc un front externe peut appeler cette API, a condition d'utiliser la bonne adresse reseau du backend.

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



utilisation api a distance lacer avec :  python local_api.py --host 0.0.0.0 --port 8765 

une requête HTTP vers :

  http://127.0.0.1:8765/api/run

  avec des fichiers dans un multipart/form-data champ files, alors ça lance le pipeline.

  Le backend exécute alors l’équivalent de :

  python main.py <fichiers_uploades> --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens

  Endpoints utiles :

  - page :

  GET http://127.0.0.1:8765/

  - lancer :

  POST http://127.0.0.1:8765/api/run

  - statut :

  GET http://127.0.0.1:8765/api/status

  Important :

  - 127.0.0.1 marche seulement si le front tourne sur la même machine que l’API
  - si le front est sur une autre machine, il faut utiliser :

  http://IP_DE_LA_MACHINE_BACK:8765

  Format attendu pour lancer :

  - POST /api/run
  - Content-Type: multipart/form-data
  - champ fichier : files
  - plusieurs fichiers possibles avec le même champ files

  Exemple JS :

  const formData = new FormData();
  formData.append("files", file1);
  formData.append("files", file2);

  const res = await fetch("http://127.0.0.1:8765/api/run", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  console.log(data);
