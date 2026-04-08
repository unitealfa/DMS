# DMS Pipeline Orchestrator

Ce dépôt regroupe des scripts de traitement documentaire (prétraitement, OCR, tokenisation, grammaire, classification) et un orchestrateur léger qui les enchaîne **sans modifier leur logique métier**.

## Architecture
- `pretraitement-de-docs.py` → `si-image-pretraiter-sinonpass-le-doc` → `output-txt.py` → `clasification.py` → `tokenisation-layout` → `atripusion-gramatical` → `table-extraction.py` → `verification-totaux.py` → `liaison-inter-docs.py` → `elasticsearch.py` → `extraction-regles.py` → `fusion_resultats.py` → `api-output.py`
- `component/tokenisation_layout/` : scripts de tokenisation/layout (`default`, `50ml`, `100ml`)
- `component/extraction/` : scripts d'extraction (`regex`, `yaml`, `50ml`, `100ml`)
- `component/fusion_resultats.py` : fichier unique de fusion pour `pipeline0ml`, `pipeline50ml`, `pipeline100ml`
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

Pour les documents stockes, le backend peut aussi renvoyer des URLs absolues publiques via:
```bash
PUBLIC_API_BASE_URL=https://mon-backend.example.com
```

Dans ce cas, `api_url` et `download_url` renvoyes au front seront des URLs absolues du type:
```text
https://mon-backend.example.com/api/documents/file/<job_id>/<filename>
```

Si `PUBLIC_API_BASE_URL` n'est pas defini, le backend utilise l'origine HTTP de la requete.

### 3) Endpoints exposes par le backend
- `GET /`
  - sert la page `index.html`
- `POST /api/run`
  - recoit les fichiers uploades, les stocke, puis lance la pipeline choisie
- `POST /api/store`
  - recoit les fichiers uploades et les stocke seulement, sans lancer la pipeline
- `GET /api/status`
  - retourne l'etat du job courant avec la vraie pipeline, le vrai composant courant et l'URL du resultat final
- `GET /api/result/<job_id>`
  - retourne le resultat final complet du job, avec le payload fusionne integral
- `GET /api/documents`
  - retourne la liste des jobs/documents stockes par l'API
- `GET /api/documents/<job_id>`
  - retourne le manifest JSON du job stocke, y compris les metadonnees du resultat API
- `GET /api/documents/file/<job_id>/<filename>`
  - retourne le fichier reel stocke par l'API
- `OPTIONS /api/run`, `OPTIONS /api/store`, `OPTIONS /api/status`
  - preflight CORS

Implementation backend: [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py)

### 4) Format exact de la requete `POST /api/run`
Content-Type requis:
- `multipart/form-data`

Champs fichier acceptes:
- `files` (recommande)
- `files[]`
- `file`

Champs texte optionnels:
- `pipeline`
  - valeurs supportees: `pipeline0ml` | `pipeline50ml` | `pipeline100ml`
- `callback_url`
  - URL HTTP distante a laquelle renvoyer le resultat final complet en `POST`
- `callback_token`
  - token Bearer ajoute dans l'entete `Authorization` du callback

Si aucun fichier n'est recu:
- reponse `400 Bad Request`

Si un job tourne deja:
- reponse `409 Conflict`

Si une erreur interne non prevue arrive apres l'upload:
- reponse `500 Internal Server Error`

Reponse normale:
- `202 Accepted`
- JSON avec:
  - `job.job_id`
  - `job.pipeline_profile`
  - `job.manifest_url`
  - `job.result_url`
  - `job.stored_documents[]`

### 5) Commande reelle lancee par le backend
Le backend construit et execute:
```bash
python main.py <fichiers_uploades_stockes> --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens
```

Si `pipeline` est envoye dans la requete, il ajoute aussi:
```bash
--pipeline pipeline0ml
```
ou:
```bash
--pipeline pipeline50ml
```
ou:
```bash
--pipeline pipeline100ml
```

Les fichiers selectionnes dans le navigateur sont d'abord copies dans un dossier dedie persistant:
```text
/home/mourad/Bureau/DMS/core/api_storage/uploads/<job_id>/
```

Puis la pipeline est lancee sur ces vrais chemins stockes dans le backend.

Fichiers generes cote backend pour un job:
- `api_storage/uploads/<job_id>/manifest.json`
- `api_storage/uploads/<job_id>/result.json`
- les fichiers reels uploades

Logs backend ajoutes pour diagnostic:
- `Content-Type`
- `Content-Length`
- champs multipart vus dans `form.list`
- nombre de fichiers extraits
- chemins absolus reellement sauvegardes
- fichiers reellement passes a `start_job(...)`

### 6) Suivi temps reel dans la page
`index.html` interroge periodiquement `GET /api/status` pour savoir si le job est:
- en cours
- termine
- en erreur

Pendant le traitement, la page affiche seulement un loader et un message simple.

Quand `status=completed`:
- la page affiche "Traitement termine"

Quand `status=failed`:
- la page affiche le `returncode` et la derniere ligne de log

### 7) Champs exacts disponibles dans `GET /api/status`
Le backend expose l'etat exact de la pipeline en cours pour un autre front/site externe.

Champs utiles:
- `pipeline_profile`
  - `pipeline0ml` | `pipeline50ml` | `pipeline100ml`
- `pipeline_source`
  - source du profil actif (`PIPELINE_DEFAULT_CODE`, `PIPELINE_DEFAULT`, `PIPELINE_PROFILE`)
- `pipeline_steps`
  - liste ordonnee exacte des composants de la pipeline active
- `pipeline_components`
  - liste detaillee des composants avec:
    - `step`
    - `component_class`
    - `script`
    - `script_path`
- `current_step`
  - composant reellement en cours ou dernier composant fini
- `component_name`
  - meme information, format explicite
- `component_script`
  - script exact utilise pour ce composant
- `component_status`
  - `running` | `completed` | `failed`
- `step_index`
  - index 1-based du composant courant dans la pipeline
- `steps_total`
  - nombre total de composants de la pipeline active
- `completed_steps_count`
  - nombre de composants deja termines
- `progress_percent`
  - avancement calcule a partir du composant reel en cours
- `last_log_line`
  - derniere ligne utile du log runtime
- `result_route`
  - route du resultat final complet
- `result_url`
  - URL complete du resultat final complet
- `result_available`
  - `true` si `result.json` est deja pret

Important:
- si la pipeline active est `pipeline100ml`, l'API renvoie les vrais composants de `pipeline100ml` et le vrai composant courant
- meme logique pour `pipeline0ml` et `pipeline50ml`
- le champ `result_available` passe a `true` seulement apres execution du composant final `api-output`

### 8) Exemple cURL
```bash
curl -X POST \
  -F "files=@documents/signettab.png" \
  -F "pipeline=pipeline100ml" \
  -F "callback_url=https://mon-site-externe.example.com/api/dms-callback" \
  http://127.0.0.1:8765/api/run
```

Puis:
```bash
curl -s http://127.0.0.1:8765/api/status
```

Puis quand `result_available=true`:
```bash
curl -s http://127.0.0.1:8765/api/result/<job_id>
```

### 9) Cycle complet de l'API
Flux reel:
1. ton site externe envoie les documents vers `POST /api/run`
2. le backend sauve les fichiers dans `api_storage/uploads/<job_id>/`
3. il cree un `manifest.json` pour ce job
4. il lance `python main.py ...`
5. l'orchestrateur construit la vraie liste des composants de la pipeline active
6. `fusion-resultats` produit le payload fusionne complet
7. le composant final `api-output` recopie ce payload integral dans `result.json`
8. si `callback_url` a ete fourni, `api-output` envoie aussi ce JSON complet en `POST` vers le site externe
9. `GET /api/status` suit l'avancement live
10. `GET /api/result/<job_id>` renvoie le JSON final complet deja pret

Donc:
- si la pipeline active est `pipeline0ml`, l'API renvoie uniquement les composants de `pipeline0ml`
- si la pipeline active est `pipeline50ml`, l'API renvoie uniquement les composants de `pipeline50ml`
- si la pipeline active est `pipeline100ml`, l'API renvoie uniquement les composants de `pipeline100ml`

### 10) Reponse type de `GET /api/status`
Exemple simplifie:
```json
{
  "status": "running",
  "job_id": "abc123",
  "pipeline_profile": "pipeline100ml",
  "pipeline_source": "PIPELINE_DEFAULT_CODE",
  "current_step": "table-extraction",
  "component_name": "table-extraction",
  "component_script": "/home/mourad/Bureau/DMS/core/component/table_extraction/table-extraction.py",
  "component_status": "running",
  "step_index": 7,
  "steps_total": 14,
  "completed_steps_count": 6,
  "progress_percent": 46,
  "result_route": "/api/result/abc123",
  "result_url": "http://127.0.0.1:8765/api/result/abc123",
  "result_available": false,
  "pipeline_steps": [
    "pretraitement-de-docs",
    "si-image-pretraiter-sinonpass-le-doc",
    "output-txt",
    "clasification",
    "tokenisation-layout",
    "atripusion-gramatical",
    "table-extraction"
  ],
  "last_log_line": "2026-04-02 ... Execution du composant table-extraction via ..."
}
```

### 11) Reponse type de `POST /api/store`
Exemple simplifie:
```json
{
  "ok": true,
  "message": "Documents stockes.",
  "job_id": "abc123",
  "pipeline_profile": "pipeline50ml",
  "storage_root": "/home/mourad/Bureau/DMS/core/api_storage/uploads",
  "manifest_route": "/api/documents/abc123",
  "manifest_url": "http://127.0.0.1:8765/api/documents/abc123",
  "result_route": "/api/result/abc123",
  "result_url": "http://127.0.0.1:8765/api/result/abc123",
  "callback_url": "https://mon-site-externe.example.com/api/dms-callback",
  "documents": [
    {
      "api_document_id": "f1",
      "file_name": "contrat.pdf",
      "content_type": "application/pdf",
      "stored_relative_path": "api_storage/uploads/abc123/contrat.pdf",
      "stored_absolute_path": "/home/mourad/Bureau/DMS/core/api_storage/uploads/abc123/contrat.pdf",
      "api_route": "/api/documents/file/abc123/contrat.pdf",
      "api_url": "https://mon-backend.example.com/api/documents/file/abc123/contrat.pdf",
      "download_url": "https://mon-backend.example.com/api/documents/file/abc123/contrat.pdf"
    }
  ]
}
```

### 12) Reponse type de `GET /api/result/<job_id>`
Ce endpoint renvoie le resultat final complet du job. Le champ `pipeline_output` contient litteralement tout le JSON fusionne produit par la pipeline, sans reduction.

Exemple simplifie:
```json
{
  "ok": true,
  "schema_version": "api-output-1.0",
  "generated_at": "2026-04-08T19:30:00+00:00",
  "job_id": "abc123",
  "pipeline_profile": "pipeline100ml",
  "pipeline_steps": [
    "pretraitement-de-docs",
    "si-image-pretraiter-sinonpass-le-doc",
    "output-txt",
    "clasification",
    "tokenisation-layout",
    "atripusion-gramatical",
    "table-extraction",
    "verification-totaux",
    "detection-signature-chachet-codebarr",
    "liaison-inter-docs",
    "elasticsearch",
    "extraction-regles",
    "fusion-resultats",
    "api-output"
  ],
  "input_documents_count": 1,
  "input_documents": [
    {
      "file_name": "contrat.pdf",
      "api_url": "https://mon-backend.example.com/api/documents/file/abc123/contrat.pdf"
    }
  ],
  "pipeline_output": {
    "schema_version": "2.0",
    "generated_at": "2026-04-08T19:29:58+00:00",
    "source": "local-context",
    "documents_count": 1,
    "documents": [
      {
        "document_id": "..."
      }
    ],
    "cross_document_analysis": {},
    "pipeline": {}
  },
  "callback_delivery": {
    "attempted": true,
    "ok": true,
    "status_code": 200
  }
}
```

Important:
- `pipeline_output` est la copie integrale du payload fusionne produit par `fusion_resultats.py`
- aucune virgule, aucun point, aucune chaine de texte extraite n'est volontairement supprimee par `api-output`
- si un champ manque pour un document, c'est la pipeline elle-meme qui ne l'a pas produit; `api-output` ne le filtre pas

### 13) Recuperer et afficher les documents depuis un autre site
Cas 1: stocker sans lancer le pipeline
- appelle `POST /api/store`
- recupere `documents[].api_url`
- utilise cette URL pour afficher ou telecharger le document

Cas 2: stocker et lancer le pipeline
- appelle `POST /api/run`
- recupere `job.stored_documents[]`
- utilise `job.stored_documents[].api_url` pour afficher les documents cote site externe
- en parallele, poll `GET /api/status` pour suivre la pipeline
- quand `result_available=true`, appelle `GET /api/result/<job_id>` pour recuperer le JSON final complet
- si `callback_url` a ete envoye, le backend poussera aussi ce JSON au site externe

Exemple JavaScript minimal:
```javascript
const API = "http://IP_DU_BACKEND:8765";

async function storeDocuments(files) {
  const formData = new FormData();
  for (const file of files) formData.append("files", file);

  const res = await fetch(`${API}/api/store`, {
    method: "POST",
    body: formData
  });
  return await res.json();
}

async function launchPipeline(files) {
  const formData = new FormData();
  for (const file of files) formData.append("files", file);
  formData.append("pipeline", "pipeline100ml");
  formData.append("callback_url", "https://mon-site-externe.example.com/api/dms-callback");

  const res = await fetch(`${API}/api/run`, {
    method: "POST",
    body: formData
  });
  return await res.json();
}

async function fetchApiResult(jobId) {
  const res = await fetch(`${API}/api/result/${jobId}`);
  return await res.json();
}

function renderDocument(url, mime) {
  if (mime === "application/pdf") {
    return `<iframe src="${url}" style="width:100%;height:700px"></iframe>`;
  }
  if ((mime || "").startsWith("image/")) {
    return `<img src="${url}" style="max-width:100%">`;
  }
  return `<a href="${url}" target="_blank">Ouvrir le document</a>`;
}
```

Pour relire un job deja stocke:
- liste globale:
```bash
curl -s http://127.0.0.1:8765/api/documents
```
- manifest d'un job:
```bash
curl -s http://127.0.0.1:8765/api/documents/<job_id>
```
- fichier reel:
```bash
curl -O http://127.0.0.1:8765/api/documents/file/<job_id>/<filename>
```
- resultat final complet:
```bash
curl -s http://127.0.0.1:8765/api/result/<job_id>
```

### 14) Recuperer ces infos depuis un autre site
Exemple JavaScript minimal:
```javascript
const API = "http://IP_DU_BACKEND:8765";

async function launchPipeline(files) {
  const formData = new FormData();
  for (const file of files) formData.append("files", file);

  const res = await fetch(`${API}/api/run`, {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Erreur lancement pipeline");
  return data;
}

async function fetchPipelineStatus() {
  const res = await fetch(`${API}/api/status`);
  return await res.json();
}

async function fetchPipelineResult(jobId) {
  const res = await fetch(`${API}/api/result/${jobId}`);
  return await res.json();
}

function watchPipeline() {
  const timer = setInterval(async () => {
    const status = await fetchPipelineStatus();

    console.log("pipeline =", status.pipeline_profile);
    console.log("etape =", status.current_step);
    console.log("composant =", status.component_name);
    console.log("script =", status.component_script);
    console.log("progression =", status.progress_percent);
    console.log("resultat_pret =", status.result_available);

    if (status.status === "completed" || status.status === "failed") {
      clearInterval(timer);
    }
  }, 1000);
}
```

Affichage conseille dans ton autre site:
- pipeline active: `pipeline_profile`
- composant en cours: `component_name`
- script exact: `component_script`
- progression: `progress_percent`
- etape courante: `current_step`
- log le plus recent: `last_log_line`
- URL du resultat final: `result_url`
- pour afficher un document stocke:
  - utilise directement `api_url`
  - ne reconstruis pas l'URL toi-meme a partir d'un chemin relatif
- pour recuperer le JSON final complet:
  - utilise `result_url`
  - ou `GET /api/result/<job_id>`

### 15) Callback sortant vers le site externe
Si le site externe envoie `callback_url`, alors le composant final `api-output` fait un `POST` JSON vers cette URL des que la pipeline est terminee.

Headers envoyes:
- `Content-Type: application/json; charset=utf-8`
- `Accept: application/json`
- `Authorization: Bearer <callback_token>` si `callback_token` est fourni

Le body du callback est exactement le meme JSON que `GET /api/result/<job_id>`.

### 16) Limite actuelle
- `GET /api/status` suit le job courant du backend local
- ce n'est pas un systeme multi-jobs paralleles
- pour un affichage live simple, il faut:
  - lancer le job avec `POST /api/run`
  - puis poller `GET /api/status` toutes les `1s` ou `1.5s`

### 17) CORS
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
