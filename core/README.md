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

## Documentation Interne
- [PROJECT_CODE_MAP.md](/home/mourad/Bureau/DMS/core/PROJECT_CODE_MAP.md)
  - cartographie technique resumee du depot
- ce `README`
  - contient maintenant aussi l'index exhaustif des fonctions/classes Python

## Guide Low-Context Pour Modifier Le Projet Sans Casser Le Reste
Cette section est ecrite pour un agent ou une API avec peu de contexte.

Objectif:
- pouvoir ajouter un composant
- pouvoir ajouter une pipeline
- pouvoir modifier l'input API
- pouvoir modifier l'output API
- sans devoir relire tout le depot

Si tu as tres peu de contexte, lis seulement ce `README` puis ouvre uniquement les fichiers cites dans cette section.

### Ordre minimal de lecture recommande
1. cette section `Guide Low-Context`
2. `## Ajout d'un nouveau composant sans retoucher tout le code`
3. `## Ajout d'une nouvelle pipeline sans retoucher le reste`
4. `## Reference Detaillee Des Pipelines`
5. `## index.html -> Backend API (detail complet)`

### Fichiers exacts a connaitre selon le type de modification
Si tu ajoutes un composant:
- [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
- [pipeline/components.py](/home/mourad/Bureau/DMS/core/pipeline/components.py)
- le script du composant dans `component/`
- optionnel: [component/fusion_resultats.py](/home/mourad/Bureau/DMS/core/component/fusion_resultats.py)
- optionnel: [component/api-output.py](/home/mourad/Bureau/DMS/core/component/api-output.py)

Si tu ajoutes une pipeline:
- [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
- optionnel: [pipeline/cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py) si tu veux changer la pipeline par defaut

Si tu modifies l'input API:
- [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py)
- optionnel: [pipeline/cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py) si le nouveau champ doit devenir une variable de contexte pour les composants
- optionnel: [index.html](/home/mourad/Bureau/DMS/core/index.html) si le front local doit aussi envoyer ce nouveau champ

Si tu modifies l'output API:
- [component/api-output.py](/home/mourad/Bureau/DMS/core/component/api-output.py)
- [dms-unified-output-template.json](/home/mourad/Bureau/DMS/core/dms-unified-output-template.json)
- optionnel: [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py) seulement si tu changes les routes API, pas si tu changes uniquement le JSON final
- optionnel: [index.html](/home/mourad/Bureau/DMS/core/index.html) si le front local doit afficher ou telecharger differemment le resultat

### Regle d'or
Si tu veux que tout continue a marcher facilement:
- garde `fusion-resultats` avant `api-output`
- garde `api-output` comme dernier composant de la pipeline
- fais ecrire les composants dans le `context`
- si une sortie est liee a un document, mets `doc_id` ou `filename`

### Recette exacte pour ajouter un composant
1. cree le fichier Python du composant dans `component/` ou un sous-dossier de `component/`
2. fais en sorte que le script lise/ecrive des variables globales du `context`
3. si le composant est simple, utilise `Component("nom", chemin_script)` dans `build_components()`
4. si le composant doit valider/enrichir quelque chose, ajoute un wrapper specialise dans [pipeline/components.py](/home/mourad/Bureau/DMS/core/pipeline/components.py)
5. ajoute ce composant dans la pipeline voulue dans [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
6. si tu veux que sa sortie apparaisse automatiquement dans le JSON final par document:
- ecris une structure contenant `doc_id`
ou
- ecris une structure contenant `filename`
7. si tu veux qu'il soit visible dans la sortie API sans mapping manuel:
- mets-le avant `api-output`
- idealement avant `fusion-resultats`

Exemple minimal de composant simple:
```python
MY_RESULT = [
  {
    "doc_id": "doc-1",
    "filename": "contrat.pdf",
    "value": "ok"
  }
]
```

Exemple minimal d'ajout dans une pipeline:
```python
Component("mon-composant", COMPONENT_DIR / "mon-composant.py")
```

### Recette exacte pour ajouter une nouvelle pipeline
1. ouvre [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
2. cree une nouvelle classe qui herite de `BasePipelineOrchestrator`
3. renseigne:
- `code`
- `aliases`
- `label`
- `description`
4. implemente `build_components()`
5. retourne une liste ordonnee de composants
6. si tu veux un comportement API standard:
- garde `FusionResultComponent("fusion-resultats", ...)`
- puis `APIOutputComponent("api-output", ...)` en dernier
7. relance `main.py` ou `local_api.py`

Exemple minimal:
```python
class Pipeline200MLOrchestrator(BasePipelineOrchestrator):
    code = "pipeline200ml"
    aliases = ("200ml",)
    label = "Pipeline 200ML"
    description = "Pipeline custom."

    def build_components(self):
        return [
            PretraitementComponent("pretraitement-de-docs", COMPONENT_DIR / "pretraitement-de-docs.py"),
            Component("mon-composant", COMPONENT_DIR / "mon-composant.py"),
            FusionResultComponent("fusion-resultats", COMPONENT_DIR / "fusion_resultats.py"),
            APIOutputComponent("api-output", COMPONENT_DIR / "api-output.py"),
        ]
```

Ce qui s'adapte automatiquement apres ca:
- `python main.py --pipeline pipeline200ml`
- `python main.py --list-steps`
- `POST /api/run` avec `pipeline=pipeline200ml`
- `GET /api/status` avec les vrais composants de cette nouvelle pipeline

### Recette exacte pour modifier l'input API
Input API = ce que le backend accepte quand un site externe appelle:
- `POST /api/run`

Fichier principal:
- [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py)

Fonctions/zones a modifier:
- `_extract_uploaded_payload(...)`
  - pour changer le format HTTP accepte
  - pour ajouter de nouveaux champs `multipart/form-data`
- `DMSLauncherHandler.do_POST(...)`
  - pour valider les nouveaux champs
  - pour les propager au job
- `LauncherState.start_job(...)`
  - pour exporter les nouveaux champs dans l'environnement du process pipeline
- [pipeline/cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py)
  - si tu veux injecter ces champs dans `context_overrides`

Chemin exact de propagation d'un champ API vers les composants:
1. le client HTTP envoie le champ a `POST /api/run`
2. `_extract_uploaded_payload()` le lit
3. `do_POST()` le valide
4. `start_job()` le convertit en variable d'environnement `DMS_API_*` ou autre
5. [pipeline/cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py) le lit et le met dans `context_overrides`
6. les composants le recuperent via le `context`

Variables API deja utilisees pour le composant final `api-output`:
- `DMS_API_JOB_ID`
- `DMS_API_MANIFEST_PATH`
- `DMS_API_RESULT_PATH`
- `DMS_API_RESULT_ROUTE`
- `DMS_API_RESULT_URL`
- `DMS_API_CALLBACK_URL`
- `DMS_API_CALLBACK_TOKEN`
- `DMS_API_CALLBACK_TIMEOUT`

Si tu ajoutes un nouveau champ API et que seul `api-output` en a besoin:
- tu peux l'ajouter dans `start_job()`
- puis le lire directement dans [component/api-output.py](/home/mourad/Bureau/DMS/core/component/api-output.py)
- sans toucher les autres composants

### Recette exacte pour modifier l'output API
Output API = JSON final renvoye par:
- `GET /api/result/<job_id>`
- et par le callback HTTP si `callback_url` est fourni

Fichiers principaux:
- [component/api-output.py](/home/mourad/Bureau/DMS/core/component/api-output.py)
- [dms-unified-output-template.json](/home/mourad/Bureau/DMS/core/dms-unified-output-template.json)

Regle actuelle:
- `api-output.py` charge d'abord `FUSION_PAYLOAD`
- il charge ensuite le template unifie
- il fusionne les donnees reelles dans le template
- si un champ manque, il reste `null` ou `[]` selon le template
- il ajoute aussi automatiquement les traces runtime des composants
- il ajoute aussi automatiquement un export global du contexte runtime exploitable par l'API
- il ecrit `result.json`
- `GET /api/result/<job_id>` sert ce `result.json`

Si tu veux changer la forme du JSON final:
1. modifie [dms-unified-output-template.json](/home/mourad/Bureau/DMS/core/dms-unified-output-template.json)
2. modifie [component/api-output.py](/home/mourad/Bureau/DMS/core/component/api-output.py) si tu veux enrichir des zones calculees comme:
- `source_context`
- `pipeline`
- `documents[].components`
- callback metadata
3. ne modifie [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py) que si tu changes les routes HTTP ou la facon de servir `result.json`

Zones les plus importantes dans `api-output.py`:
- `_load_fusion_payload(...)`
  - source brute a normaliser
- `_load_template_payload(...)`
  - lecture du template cible
- `_merge_template(...)`
  - fusion template + donnees reelles
- `_normalize_for_api(...)`
  - enrichissement du resultat final
- `_overlay_component_traces(...)`
  - ajout automatique des sorties des nouveaux composants

Ce que l'API finale renvoie maintenant automatiquement pour eviter d'avoir a modifier la partie API a chaque nouveau composant:
- `pipeline.component_runs[]`
  - une ligne par composant execute
  - contient le `reported_output`
  - contient aussi `context_values`
- `pipeline.context_exports`
  - export global JSON-safe des cles de contexte utiles du run
  - permet de recuperer automatiquement des structures comme:
    - `FINAL_DOCS`
    - `RESULTS`
    - `TOK_DOCS`
    - `NLP_SENTENCES`
    - `NLP_TOKENS`
    - `NLP_ENTITIES`
    - `TABLE_EXTRACTIONS`
    - `TOTALS_VERIFICATION`
    - `EXTRACTIONS`
    - et les autres sorties futures exposees en contexte

Regle pratique:
- si tu ajoutes un nouveau composant qui ecrit ses donnees dans le `context`, l'API finale les renverra automatiquement via:
  - `documents[].components.<nom_du_composant>`
  - ou `pipeline.component_runs[].context_values`
  - ou `pipeline.context_exports`
- donc tu n'as pas besoin de reprogrammer `local_api.py` ou `api-output.py` pour chaque nouveau composant standard

### Cas ou il ne faut pas modifier autre chose
Si tu ajoutes seulement un composant standard:
- modifie seulement le script du composant
- puis [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)

Si tu ajoutes seulement une pipeline:
- modifie seulement [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
- optionnel: [pipeline/cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py) pour changer la pipeline par defaut

Si tu changes seulement la structure du JSON final:
- modifie seulement [dms-unified-output-template.json](/home/mourad/Bureau/DMS/core/dms-unified-output-template.json)
- et si necessaire [component/api-output.py](/home/mourad/Bureau/DMS/core/component/api-output.py)

Si tu changes seulement le format de requete HTTP d'entree:
- modifie seulement [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py)
- et [index.html](/home/mourad/Bureau/DMS/core/index.html) si le front local doit suivre

### Validation minimale apres modification
Pour un composant:
```bash
python main.py --pipeline pipeline50ml --list-steps
```

Pour une pipeline:
```bash
python main.py documents/englais.docx --pipeline <nouveau_code> --upto api-output
```

Pour l'API:
```bash
python local_api.py --host 0.0.0.0 --port 8765
```

Puis:
```bash
curl -X POST -F "files=@documents/englais.docx" -F "pipeline=<nouveau_code>" http://127.0.0.1:8765/api/run
```

Puis:
```bash
curl -s http://127.0.0.1:8765/api/status
```

Puis:
```bash
curl -s http://127.0.0.1:8765/api/result/<job_id>
```

### Reponse courte a la question "est-ce qu'une API peu puissante peut s'en sortir avec seulement le README ?"
Oui, maintenant, si elle suit strictement cette section et les sections juste en dessous.

## Ajout d'un nouveau composant sans retoucher tout le code
Le pipeline est maintenant prepare pour qu'un nouveau composant s'integre sans devoir modifier:
- le schema de sortie final
- les listes de steps CLI
- le resultat API final

### Regle pratique
Pour ajouter un nouveau composant simple:
1. creer le script Python du composant dans `component/`
2. ajouter ce composant dans la pipeline voulue dans [orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
3. utiliser directement le wrapper generique `Component(...)` si aucune logique speciale n'est necessaire

Exemple:
```python
Component("mon-nouveau-composant", COMPONENT_DIR / "mon-nouveau-composant.py")
```

Le wrapper generique execute le script, trace automatiquement:
- les nouvelles cles ajoutees au `context`
- les cles modifiees
- le script utilise
- le statut du composant

### Ce qui s'adapte automatiquement
- `main.py --list-steps`
  - les steps sont maintenant calcules dynamiquement depuis les orchestrateurs
- `--only`, `--upto`, `--start`
  - les nouvelles etapes deviennent disponibles automatiquement
- `fusion_resultats.py`
  - les sorties de composants non explicitement mappes sont exposees automatiquement dans `documents[].components.<nom_du_composant>`
- `api-output.py`
  - le resultat API final reprend aussi automatiquement les traces des nouveaux composants
- `GET /api/status`
  - la liste exacte des composants de la pipeline active est reconstruite depuis l'orchestrateur reel

### Ce que le composant doit faire pour etre auto-pris en charge
Le script du composant doit simplement ecrire ses donnees dans le dictionnaire global partage (`context` / `init_globals`).

Exemple minimal:
```python
MY_NEW_RESULT = [
  {"filename": "document.pdf", "doc_id": "doc-1", "value": "ok"}
]
```

Le wrapper detecte automatiquement les cles touchees et les rend visibles dans:
- `pipeline.component_runs`
- `documents[].components.<nom_du_composant>`

### Correspondance document automatique
Pour qu'une sortie soit rattachee automatiquement au bon document dans le resultat final, la valeur produite par le composant doit idealement contenir:
- `doc_id`
ou
- `filename`

Si la pipeline ne traite qu'un seul document, le systeme peut aussi rattacher automatiquement certaines sorties simples sans `doc_id`.

### Quand utiliser un wrapper specifique
Les wrappers specialises existants (`PretraitementComponent`, `OutputTxtComponent`, `FusionResultComponent`, etc.) restent utiles quand il faut:
- valider une sortie obligatoire
- enrichir le resume terminal
- convertir/reconcilier des cles amont/aval

Mais pour un composant standard, il n'est plus necessaire de creer une nouvelle classe dans `pipeline/components.py`.

### Limite precise
Si un nouveau composant est ajoute avant `fusion-resultats`, alors:
- `fusion_output.json` peut l'exposer automatiquement via `documents[].components`
- `result.json` de l'API l'expose aussi

Si un nouveau composant est ajoute apres `fusion-resultats`, alors:
- `fusion_output.json` ne peut pas etre retro-injecte dans ce run
- mais `api-output.py` recompose quand meme le resultat API final avec les traces runtime du composant

### Fichiers qui portent ce mecanisme
- [pipeline/component_trace.py](/home/mourad/Bureau/DMS/core/pipeline/component_trace.py)
- [pipeline/components.py](/home/mourad/Bureau/DMS/core/pipeline/components.py)
- [pipeline/cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py)
- [component/fusion_resultats.py](/home/mourad/Bureau/DMS/core/component/fusion_resultats.py)
- [component/api-output.py](/home/mourad/Bureau/DMS/core/component/api-output.py)

## Ajout d'une nouvelle pipeline sans retoucher le reste
Le systeme est maintenant aussi prepare pour qu'une nouvelle pipeline soit detectee automatiquement par:
- la CLI
- `--list-steps`
- `--only`, `--upto`, `--start`
- l'API locale
- `GET /api/status`
- le champ `pipeline` de `POST /api/run`

### Regle pratique
Pour ajouter une nouvelle pipeline:
1. ouvrir [orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
2. creer une nouvelle classe qui herite de `BasePipelineOrchestrator`
3. definir au minimum:
   - `code`
   - `aliases` si necessaire
   - `label`
   - `description`
   - `build_components()`
4. mettre les composants voulus dans `build_components()`
5. si tu veux que `default` pointe dessus, mettre ce code dans `PIPELINE_DEFAULT_CODE` de [cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py)

Exemple minimal:
```python
class Pipeline200MLOrchestrator(BasePipelineOrchestrator):
    code = "pipeline200ml"
    aliases = ("200ml",)
    label = "Pipeline 200ML"
    description = "Pipeline custom."

    def build_components(self):
        return [
            PretraitementComponent("pretraitement-de-docs", COMPONENT_DIR / "pretraitement-de-docs.py"),
            Component("mon-composant", COMPONENT_DIR / "mon-composant.py"),
            APIOutputComponent("api-output", COMPONENT_DIR / "api-output.py"),
        ]
```

### Ce qui s'adapte automatiquement
- la liste des pipelines disponibles est reconstruite dynamiquement depuis les sous-classes de `BasePipelineOrchestrator`
- la normalisation des noms/alias de pipeline est dynamique
- `default` pointe automatiquement vers la valeur actuelle de `PIPELINE_DEFAULT_CODE`
- `python main.py --pipeline <nouveau_code>` marche sans ajouter de `if/elif`
- `python main.py --list-steps` voit la nouvelle pipeline
- `_step_choices()` de la CLI voit aussi ses composants
- `local_api.py` expose automatiquement:
  - `pipeline_profile`
  - `pipeline_label`
  - `pipeline_description`
  - `pipeline_steps`
  - `pipeline_components`

### Fichiers qui portent ce mecanisme pipeline
- [pipeline/orchestrator.py](/home/mourad/Bureau/DMS/core/pipeline/orchestrator.py)
- [pipeline/cli.py](/home/mourad/Bureau/DMS/core/pipeline/cli.py)
- [pipeline/local_api.py](/home/mourad/Bureau/DMS/core/pipeline/local_api.py)

### Valeurs acceptees par `POST /api/run`
Le champ `pipeline` accepte maintenant:
- n'importe quel `code` de pipeline enregistre
- ou un `alias` defini sur cette pipeline
- ou `default`

Exemple:
```bash
curl -X POST \
  -F "files=@documents/signettab.png" \
  -F "pipeline=pipeline200ml" \
  http://127.0.0.1:8765/api/run
```

### Limite precise
La nouvelle pipeline doit quand meme:
- etre importable au demarrage du process
- etre definie avant que la CLI ou l'API locale ne demarre

En pratique:
- si tu ajoutes la classe dans `pipeline/orchestrator.py`, c'est bon
- ensuite tu peux juste relancer `main.py` ou `local_api.py`

## Reference Detaillee Des Pipelines

### But
- ce bloc explique quelle pipeline se lance, dans quel ordre, et ce que chaque composant produit
- il decrit l'etat reel du code actuel

### Choix de la pipeline
- fichier a regarder: `pipeline/cli.py`
- variable actuelle par defaut: `PIPELINE_DEFAULT_CODE = "pipeline50ml"`
- `default` pointe vers la valeur de `PIPELINE_DEFAULT_CODE`
- la selection n'est plus faite par une liste statique codee en dur
- les pipelines sont auto-decouvertes depuis `pipeline/orchestrator.py`

### Comment le code choisit la pipeline
- `pipeline/cli.py` normalise le nom demande
- puis appelle `create_pipeline_orchestrator(...)`
- ce helper lit le registre dynamique des pipelines dans `pipeline/orchestrator.py`
- `local_api.py` utilise exactement la meme logique

### Ce qu'il faut modifier si tu veux changer la pipeline par defaut
- fichier: `pipeline/cli.py`
- variable: `PIPELINE_DEFAULT_CODE`

### Ce qu'il faut modifier si tu veux ajouter une nouvelle pipeline
- fichier: `pipeline/orchestrator.py`
- creer une nouvelle classe qui herite de `BasePipelineOrchestrator`
- definir:
  - `code`
  - `aliases` si besoin
  - `label`
  - `description`
  - `build_components()`

### Pipelines actuellement enregistrees
- `pipeline0ml`
- `pipeline50ml`
- `pipeline100ml`

### Pipeline `pipeline0ml`
`self.components = [`
- `PretraitementComponent("pretraitement-de-docs", "component/pretraitement-de-docs.py")`
- `OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", "component/si-image-pretraiter-sinonpass-le-doc.py")`
- `OutputTxtComponent("output-txt", "component/output-txt.py")`
- `ClassificationComponent("clasification", "component/clasification.py")`
- `TokenisationLayoutComponent("tokenisation-layout", "component/tokenisation_layout/tokenisation-layout.py")`
- `GrammarComponent("atripusion-gramatical", "component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py")`
- `TableExtractionComponent("table-extraction", "component/table_extraction/table-extraction.py")`
- `TotalsVerificationComponent("verification-totaux", "component/verification-totaux.py")`
- `InterDocLinkingComponent("liaison-inter-docs", "component/liaison-inter-docs.py")`
- `ElasticsearchComponent("elasticsearch", "component/elasticsearch.py")`
- `RuleExtractionComponent("extraction-regles", "component/extraction/extraction-regles.py")`
- `FusionResultComponent("fusion-resultats", "component/fusion_resultats.py")`
- `APIOutputComponent("api-output", "component/api-output.py")`
`]`

### Pipeline `pipeline50ml`
`self.components = [`
- `PretraitementComponent("pretraitement-de-docs", "component/pretraitement-de-docs.py")`
- `OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", "component/si-image-pretraiter-sinonpass-le-doc.py")`
- `OutputTxtComponent("output-txt", "component/output-txt.py")`
- `ClassificationComponent("clasification", "component/clasification.py")`
- `TokenisationLayoutComponent("tokenisation-layout", "component/tokenisation_layout/tokenisation-layout-50ml.py")`
- `GrammarComponent("atripusion-gramatical", "component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py")`
- `TableExtractionComponent("table-extraction", "component/table_extraction/table-extraction.py")`
- `TotalsVerificationComponent("verification-totaux", "component/verification-totaux.py")`
- `InterDocLinkingComponent("liaison-inter-docs", "component/liaison-inter-docs.py")`
- `ElasticsearchComponent("elasticsearch", "component/elasticsearch.py")`
- `RuleExtractionComponent("extraction-regles", "component/extraction/extraction-regles-50ml.py")`
- `FusionResultComponent("fusion-resultats", "component/fusion_resultats.py")`
- `APIOutputComponent("api-output", "component/api-output.py")`
`]`

### Pipeline `pipeline100ml`
`self.components = [`
- `PretraitementComponent("pretraitement-de-docs", "component/pretraitement-de-docs.py")`
- `OCRPreprocessComponent("si-image-pretraiter-sinonpass-le-doc", "component/si-image-pretraiter-sinonpass-le-doc.py")`
- `OutputTxtComponent("output-txt", "component/output-txt.py")`
- `ClassificationComponent("clasification", "component/clasification.py")`
- `TokenisationLayoutComponent("tokenisation-layout", "component/tokenisation_layout/tokenisation-layout-100ml.py")`
- `GrammarComponent("atripusion-gramatical", "component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py")`
- `TableExtractionComponent("table-extraction", "component/table_extraction/table-extraction.py")`
- `TotalsVerificationComponent("verification-totaux", "component/verification-totaux.py")`
- `VisualMarksDetectionComponent("detection-signature-chachet-codebarr", "component/detection-signature-chachet-codebarr.py")`
- `InterDocLinkingComponent("liaison-inter-docs", "component/liaison-inter-docs.py")`
- `ElasticsearchComponent("elasticsearch", "component/elasticsearch.py")`
- `RuleExtractionComponent("extraction-regles", "component/extraction/extraction-regles-100ml.py")`
- `FusionResultComponent("fusion-resultats", "component/fusion_resultats.py")`
- `APIOutputComponent("api-output", "component/api-output.py")`
`]`

### Comment les composants communiquent
- le pipeline partage un dictionnaire `context`
- entree initiale: `INPUT_FILE`
- chaque composant lit le `context`, ecrit ses sorties, puis le composant suivant reutilise ces donnees
- l'execution se fait via `runpy.run_path(..., init_globals=context)` dans `pipeline/components.py`

### Flux logique principal
- fichier -> pretraitement -> routage texte/OCR -> extraction texte -> classification -> tokenisation/layout -> grammaire/NLP -> tableaux -> verification totaux -> visuel 100ml -> liaisons inter-docs -> Elasticsearch -> extraction metier -> fusion finale -> sortie API

### Cles de contexte les plus importantes
- `INPUT_FILE`
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
- `API_OUTPUT_RESULT`

### Ce que chaque grand composant fait

#### 1) `pretraitement-de-docs`
- detecte le format
- classe grossierement le contenu (`text`, `image_only`, `unsupported`)
- sortie principale: `PRETRAITEMENT_RESULT`

#### 2) `si-image-pretraiter-sinonpass-le-doc`
- separe les fichiers texte natifs des fichiers image/OCR
- sortie principale: `TEXT_FILES`, `IMAGE_ONLY_FILES`, `DOCS`

#### 3) `output-txt`
- produit le texte final exploitable
- gere PDF, DOCX, HTML, images OCR, XLSX
- sortie principale: `FINAL_DOCS`

#### 4) `clasification`
- attribue un type documentaire
- sortie principale: `RESULTS`

#### 5) `tokenisation-layout`
- construit pages, chunks, phrases et structure layout
- `pipeline50ml` ajoute des vecteurs/hash/topics legers
- `pipeline100ml` ajoute des embeddings Transformer et topics document/chunk
- sortie principale: `TOK_DOCS`

#### 6) `atripusion-gramatical`
- produit POS, lemmas, NER, langues
- version standard pour `pipeline0ml` et `pipeline50ml`
- version XLM-R pour `pipeline100ml`
- sorties principales: `NLP_ANALYSES`, `NLP_SENTENCES`, `NLP_TOKENS`, `NLP_ENTITIES`

#### 7) `table-extraction`
- detecte les tableaux
- reconstruit lignes utiles produit/quantite/prix/total
- sorties principales: `TABLE_EXTRACTIONS`, `TABLE_EXTRACTIONS_50ML`, `TABLE_EXTRACTIONS_100ML`

#### 8) `verification-totaux`
- verifie la coherence des lignes et des totaux
- sortie principale: `TOTALS_VERIFICATION`

#### 9) `detection-signature-chachet-codebarr`
- uniquement dans `pipeline100ml`
- detecte signatures, cachets, QR et codes-barres
- sorties principales: `VISUAL_MARKS_DETECTIONS`, `VISUAL_MARKS_DETECTIONS_100ML`

#### 10) `liaison-inter-docs`
- compare les documents entre eux
- detecte liens par topics, phrases et vecteurs si disponibles
- sorties principales: `INTERDOC_ANALYSIS`, `INTERDOC_LINKS`

#### 11) `elasticsearch`
- indexe et relit des donnees si ES est actif
- sorties principales: `ES_*`

#### 12) `extraction-regles`
- applique les regles metier
- version standard pour `pipeline0ml`
- versions YAML/BM25 pour `pipeline50ml` et `pipeline100ml`
- sortie principale: `EXTRACTIONS`

#### 13) `fusion-resultats`
- fusionne toutes les sorties dans `fusion_output.json`
- sorties principales: `FUSION_RESULT`, `FUSION_PAYLOAD`, `FUSION_PAYLOADS`

#### 14) `api-output`
- normalise la sortie finale sur le template API unifie
- ecrit `result.json` pour l'API locale
- sortie principale: `API_OUTPUT_RESULT`

### Technos importantes vraiment utilisees
- OCR: `Tesseract`
- embeddings 100ml: `transformers` + `xlm-roberta-base`
- pipeline 50ml: mode `fasttext-like` local par hashing subword
- BM25: seulement dans `extraction-regles-50ml.py` et `extraction-regles-100ml.py`
- Elasticsearch: optionnel selon la config/runtime
- Word2Vec: non utilise

### Fichiers a retenir
- choix par defaut pipeline: `pipeline/cli.py`
- registre et definition des pipelines: `pipeline/orchestrator.py`
- wrappers des composants: `pipeline/components.py`
- etat runtime API: `pipeline/local_api.py`
- sortie terminal: `outputgeneralterminal.txt`
- sortie fusion debug: `fusion_output.json`
- sortie API finale runtime: `/tmp/dms_api_runtime/<job_id>/result.json`
- sortie API consommee par un client externe: `GET /api/result/<job_id>`

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

## Commandes De Lancement Utiles
Cette section remplace entierement l'ancien mémo `commade lacer le code.txt`.
Objectif conserve:
- structurer toutes les informations des documents dans une sortie JSON

### Activer l'environnement
```bash
source .venv/bin/activate
```

### Jeu de documents exemple
Le mémo d'origine utilisait cette liste de travail:
```bash
DOCS_SAMPLE="documents/testword.docx,documents/testexcel.xlsx,documents/signettab.png,documents/image2tab.webp,documents/francais.docx,documents/arab.docx,documents/englais.docx,documents/testexcel.xlsx"
```

### Lancer et s'arreter a une etape precise
Equivalent actuel corrige du mémo historique:

1. Prétraitement seul
```bash
python main.py "$DOCS_SAMPLE" --upto pretraitement-de-docs
```

2. Prétraitement + routage OCR/natif
```bash
python main.py "$DOCS_SAMPLE" --upto si-image-pretraiter-sinonpass-le-doc
```

3. Jusqu'à l'extraction texte
```bash
python main.py "$DOCS_SAMPLE" --upto output-txt
```

4. Jusqu'à la classification
```bash
python main.py "$DOCS_SAMPLE" --upto clasification
```

5. Jusqu'à la tokenisation/layout
```bash
python main.py "$DOCS_SAMPLE" --upto tokenisation-layout
```

6. Jusqu'à la grammaire/NLP
```bash
python main.py "$DOCS_SAMPLE" --upto atripusion-gramatical
```

7. Pipeline complet
```bash
python main.py "$DOCS_SAMPLE"
```

### Variante package
Si tu veux utiliser le script package au lieu de `main.py`:
```bash
orchestre "$DOCS_SAMPLE" --upto output-txt
```

### Mode Elasticsearch
Si Elasticsearch doit etre demarre manuellement:
```bash
sudo systemctl start elasticsearch
sudo systemctl status elasticsearch --no-pager
curl -s http://localhost:9200
```

#### Stocker la grammaire complete dans Elasticsearch
```bash
python main.py documents/image2tab.webp --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens
python main.py documents/contrat_regex_test_corpus_fr_en_ar.pdf --use-elasticsearch --es-nlp-level full --es-nlp-index dms_nlp_tokens
```

#### Ne stocker que le résumé NLP
```bash
python main.py documents/contrat_regex_test_corpus_fr_en_ar.pdf --use-elasticsearch --es-nlp-level summary
```

#### Ne rien stocker côté NLP dans Elasticsearch
```bash
python main.py documents/contrat_regex_test_corpus_fr_en_ar.pdf --use-elasticsearch --es-nlp-level off
```

### Lancer l'API locale
Commande normale:
```bash
python local_api.py --host 0.0.0.0 --port 8765
```

Commande arrière-plan:
```bash
nohup python local_api.py --host 0.0.0.0 --port 8765 > local_api.log 2>&1 &
```

### Tuer le process qui occupe le port 8765
```bash
kill -9 $(lsof -t -i :8765)
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
  - recoit les fichiers uploades, les materialise temporairement en runtime, puis lance la pipeline choisie
- `GET /api/status`
  - retourne l'etat du job courant avec la vraie pipeline, le vrai composant courant et l'URL du resultat final
- `GET /api/result/<job_id>`
  - retourne le resultat final complet du job, avec le payload fusionne integral
- `GET /api/documents`
  - retourne la liste des jobs/documents encore connus en memoire par l'API
- `GET /api/documents/<job_id>`
  - retourne le manifest JSON du job, y compris les metadonnees du resultat API
- `GET /api/documents/file/<job_id>/<filename>`
  - retourne le fichier temporaire du job seulement tant qu'il existe encore en runtime
- `OPTIONS /api/run`, `OPTIONS /api/status`
  - preflight CORS

Important:
- `POST /api/store` est desactive
- les fichiers envoyes a `POST /api/run` ne sont plus stockes de facon persistante dans le depot
- le mode actuel est `temporary-no-persistence`
- le backend utilise un dossier runtime temporaire, puis nettoie ce dossier apres la fin du job
- les fichiers d'entree sont supprimes du disque a la fin du job
- le resultat JSON final n'est garde qu'en memoire le temps de la livraison
- si `callback_url` est fourni et que le callback reussit, le resultat est purge de la memoire juste apres envoi
- si aucun callback n'est fourni, le resultat est purge de la memoire juste apres le premier `GET /api/result/<job_id>`
- donc `GET /api/result/<job_id>` doit etre considere comme une lecture de livraison, pas comme un stockage permanent

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
  - accepte tout `code` de pipeline enregistre
  - accepte aussi les `aliases` declares par la pipeline
  - accepte aussi `default`
  - pipelines actuellement presentes dans le depot: `pipeline0ml`, `pipeline50ml`, `pipeline100ml`
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
--pipeline <code_pipeline>
```

Exemples actuels:
- `--pipeline pipeline0ml`
- `--pipeline pipeline50ml`
- `--pipeline pipeline100ml`

Les fichiers selectionnes dans le navigateur sont d'abord materialises dans un dossier runtime temporaire:
```text
/tmp/dms_api_runtime/<job_id>/inputs/
```

Puis la pipeline est lancee sur ces vrais chemins temporaires dans le backend.

Fichiers generes cote backend pour un job:
- `/tmp/dms_api_runtime/<job_id>/manifest.json`
- `/tmp/dms_api_runtime/<job_id>/result.json`
- les fichiers reels uploades sous `/tmp/dms_api_runtime/<job_id>/inputs/`

Important:
- ces fichiers runtime ne sont pas la destination finale du systeme
- ils servent seulement a executer le job en cours
- apres la fin du job, le runtime disque est nettoye automatiquement
- la reponse de `GET /api/result/<job_id>` continue de fonctionner grace au cache memoire du backend

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

`index.html` utilise seulement l'API HTTP du backend. La page ne lit pas directement les fichiers du job ni les sorties Python internes. Elle fait uniquement:
- `POST /api/run`
- `GET /api/status`
- `GET /api/result/<job_id>`

Pendant le traitement, la page affiche seulement:
- un loader
- un message simple

Comportement exact de la page:
1. verification que `GET /api/status` repond
2. envoi des fichiers via `POST /api/run`
3. memorisation de `job_id`
4. memorisation de `result_url` ou `result_route`
5. polling de `GET /api/status` toutes les `1.5s`
6. si `status=running`, la page garde le loader et affiche l'etape courante si disponible
7. si `status=completed` mais `result_available=false`, la page attend encore
8. si `status=completed` et `result_available=true`, la page appelle `GET /api/result/<job_id>`
9. la reponse JSON est transformee en fichier telecharge automatiquement dans le navigateur
10. le nom du fichier telecharge est:
```text
dms-output-<job_id>.json
```

La page ne depend plus d'une version API exacte hardcodee. Tant que les endpoints API repondent correctement, elle fonctionne.

Quand `status=completed`:
- la page telecharge automatiquement le resultat JSON final
- puis elle affiche "Traitement termine"

Quand `status=failed`:
- la page affiche le `returncode` et la derniere ligne de log

### 7) Champs exacts disponibles dans `GET /api/status`
Le backend expose l'etat exact de la pipeline en cours pour un autre front/site externe.

Champs utiles:
- `pipeline_profile`
  - code exact de la pipeline active
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
- si la pipeline active est une pipeline custom enregistree dans `pipeline/orchestrator.py`, l'API renvoie aussi ses vrais composants et son vrai composant courant
- meme logique pour les pipelines deja presentes dans le depot
- le champ `result_available` passe a `true` seulement apres execution du composant final `api-output`
- une fois le resultat livre, `result_available` repasse a `false`
- les champs `result_delivered`, `result_delivery_mode`, `result_delivered_at` et `artifacts_purged` permettent de savoir si le payload a deja ete remis puis purge

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
2. le backend materialise les fichiers dans un runtime temporaire `/tmp/dms_api_runtime/<job_id>/inputs/`
3. il cree un `manifest.json` pour ce job
4. il lance `python main.py ...`
5. l'orchestrateur construit la vraie liste des composants de la pipeline active
6. `fusion-resultats` produit le payload fusionne complet
7. le composant final `api-output` recopie ce payload integral dans `result.json`
8. `api-output` enrichit aussi automatiquement le resultat avec:
   - `pipeline.component_runs[]`
   - `pipeline.component_runs[].reported_output`
   - `pipeline.component_runs[].context_values`
   - `pipeline.component_runs[].stdout_text`
   - `pipeline.component_runs[].stdout_lines`
   - `pipeline.component_runs[].stderr_text`
   - `pipeline.component_runs[].stderr_lines`
   - `pipeline.component_runs[].report_text`
   - `pipeline.component_runs[].report_lines`
   - `pipeline.component_runs[].terminal_text`
   - `pipeline.component_runs[].terminal_lines`
   - `pipeline.context_exports`
   - `pipeline.terminal_text`
   - `pipeline.terminal_lines`
   - `documents[].components.<component_key>.data`
   - `documents[].components.<component_key>.stdout_text`
   - `documents[].components.<component_key>.stderr_text`
   - `documents[].components.<component_key>.terminal_text`
   - et les variantes `*_lines`
9. si `callback_url` a ete fourni, `api-output` envoie aussi ce JSON complet en `POST` vers le site externe
10. `GET /api/status` suit l'avancement live
11. `GET /api/result/<job_id>` renvoie le JSON final complet deja pret
12. le runtime disque du job est nettoye automatiquement
13. si le callback reussit, le resultat en memoire est purge juste apres livraison
14. sinon, le resultat en memoire est purge apres le premier `GET /api/result/<job_id>`

Donc:
- si la pipeline active est `pipeline0ml`, l'API renvoie uniquement les composants de `pipeline0ml`
- si la pipeline active est `pipeline50ml`, l'API renvoie uniquement les composants de `pipeline50ml`
- si la pipeline active est `pipeline100ml`, l'API renvoie uniquement les composants de `pipeline100ml`
- si tu ajoutes une nouvelle pipeline enregistree, l'API renvoie aussi automatiquement uniquement les composants de cette nouvelle pipeline

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
  "result_delivered": false,
  "result_delivery_mode": null,
  "result_delivered_at": null,
  "artifacts_purged": false,
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

### 11) Note importante sur `POST /api/store`
Cette route est desactivee.

Reponse actuelle:
```json
{
  "error": "POST /api/store est desactive. Les documents ne sont plus stockes de facon persistante."
}
```

### 12) Reponse type de `GET /api/result/<job_id>`
Ce endpoint renvoie directement le resultat final du job au format du template unifie:
- [dms-unified-output-template.json](/home/mourad/Bureau/DMS/core/dms-unified-output-template.json)

Regle appliquee:
- si une donnee existe dans la pipeline, elle est injectee
- si elle manque, la valeur reste `null`
- les listes absentes restent `[]`
- les objets restent materialises selon le template
- la structure racine suit le template unifie
- `schema_version` vient du template API
- `source_context.fusion_schema_version` garde la version du payload fusionne brut
- si un champ reel existe dans `fusion_resultats.py` mais n'est pas explicitement decrit dans le template, `api-output` le preserve quand meme au bon endroit
- pour une liste d'objets, chaque element est normalise contre le template du premier element de cette liste

Exemple simplifie:
```json
{
  "schema_version": "dms-unified-final-template-1.0",
  "generated_at": "2026-04-08T19:30:00+00:00",
  "source": "local-context",
  "profile": "pipeline100ml",
  "documents_count": 1,
  "documents": [
    {
      "document_id": "...",
      "file": {
        "name": "contrat.pdf",
        "paths": [
          "/tmp/dms_api_runtime/abc123/inputs/contrat.pdf"
        ],
        "size": 12345,
        "page_count": 14,
        "mime": "application/pdf",
        "ext": ".pdf",
        "content_mode": "text"
      },
      "classification": {
        "doc_type": "CONTRAT",
        "winning_score": 11
      },
      "ml0": {
        "table_extraction": {
          "engine": null
        }
      },
      "ml50": {
        "embedding_method": null
      },
      "ml100": {
        "embedding_method": "transformer"
      }
    }
  ],
  "cross_document_analysis": {
    "links_count": 0
  },
  "pipeline": {
    "profile": "pipeline100ml",
    "component_runs": [
      {
        "component_name": "output-txt",
        "component_script": "component/output-txt.py",
        "status": "completed",
        "summary": "1 docs | pages=14",
        "reported_output_type": "list",
        "reported_output": [
          {
            "doc_id": "...",
            "filename": "contrat.pdf"
          }
        ],
        "context_values": {
          "FINAL_DOCS": [
            {
              "doc_id": "...",
              "filename": "contrat.pdf"
            }
          ]
        },
        "stdout_text": "Texte extrait...\\nAutre ligne...\\n",
        "stdout_lines": [
          "Texte extrait...",
          "Autre ligne..."
        ],
        "stderr_text": "",
        "stderr_lines": [],
        "report_text": "[Component: output-txt]\\nType: list\\nSummary: 1 docs | pages=14\\nOutput: ...",
        "report_lines": [
          "[Component: output-txt]",
          "Type: list",
          "Summary: 1 docs | pages=14"
        ],
        "terminal_text": "Execution du composant output-txt via ...\\nTexte extrait...\\n[Component: output-txt]\\nType: list\\nSummary: 1 docs | pages=14\\nOutput: ...",
        "terminal_lines": [
          "Execution du composant output-txt via ...",
          "Texte extrait...",
          "[Component: output-txt]",
          "Type: list",
          "Summary: 1 docs | pages=14"
        ]
      }
    ],
    "context_exports": {
      "FINAL_DOCS": [
        {
          "doc_id": "...",
          "filename": "contrat.pdf"
        }
      ]
    },
    "terminal_text": "Execution du composant pretraitement-de-docs via ...\\n...",
    "terminal_lines": [
      "Execution du composant pretraitement-de-docs via ...",
      "..."
    ]
  },
  "source_context": {
    "input_files": [
      "/tmp/dms_api_runtime/abc123/inputs/contrat.pdf"
    ],
    "source_mode": "api",
    "fusion_schema_version": "2.0",
    "profile_requested": "pipeline100ml",
    "profile_effective": "pipeline100ml"
  }
}
```

Important:
- la reponse de `GET /api/result/<job_id>` ressemble au template unifie final
- `api-output` part du template puis y injecte les donnees reelles du job
- aucune chaine de texte extraite n'est volontairement supprimee
- si un champ manque pour un document, il reste `null` dans la sortie finale
- les sorties terminal des composants sont maintenant exposees explicitement en JSON
- si un composant ecrit sur `stdout` ou `stderr`, cette sortie remonte automatiquement
- si un composant retourne un objet Python, il remonte automatiquement dans `reported_output`
- si un composant touche des cles du `context`, ces cles remontent automatiquement dans `context_values`
- si un nouveau composant standard est ajoute a une pipeline, son `reported_output`, ses `context_values` et sa trace terminal remontent automatiquement sans modifier `local_api.py`
- quand `index.html` recoit ce resultat final, elle le telecharge directement en `.json`
- apres livraison du resultat, le backend ne conserve plus le payload complet en memoire
- donc si le client veut archiver le JSON, c'est au client externe de le stocker

### 13) Recuperer et afficher les documents depuis un autre site
Cas pratique:
- appelle `POST /api/run`
- recupere `job.stored_documents[]`
- si le job est encore en cours, `job.stored_documents[].api_url` peut servir pour afficher le document temporaire cote site externe
- une fois le job termine, le runtime disque est nettoye, donc cette URL de fichier temporaire n'est plus garantie
- la vraie source stable a consommer cote site externe est `GET /api/result/<job_id>` ou le callback JSON final
- si `callback_url` est configure et reussit, il faut consommer le resultat dans ce callback, pas attendre un `GET /api/result/<job_id>`
- si aucun callback n'est configure, le client externe doit lire `GET /api/result/<job_id>` une seule fois puis stocker lui-meme le JSON s'il en a besoin
- en parallele, poll `GET /api/status` pour suivre la pipeline
- quand `result_available=true`, appelle `GET /api/result/<job_id>` pour recuperer le JSON final complet normalise sur le template
- si `callback_url` a ete envoye, le backend poussera aussi ce JSON au site externe

Exemple JavaScript minimal:
```javascript
const API = "http://IP_DU_BACKEND:8765";

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

Le body du callback est exactement le meme JSON que `GET /api/result/<job_id>`, donc au format du template unifie final.

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

## Index Exhaustif Des Fonctions Et Classes

Le contenu ci-dessous remplace entierement l'ancien `FUNCTION_INDEX.txt`.
Il conserve chaque entree `fichier:ligne | type | nom`.

```text
FUNCTION_INDEX

component/api-output.py:18 | FunctionDef | _iso_now
component/api-output.py:22 | FunctionDef | _safe_dict
component/api-output.py:26 | FunctionDef | _safe_list
component/api-output.py:30 | FunctionDef | _read_json
component/api-output.py:42 | FunctionDef | _write_json
component/api-output.py:51 | FunctionDef | _load_fusion_payload
component/api-output.py:59 | FunctionDef | _load_template_payload
component/api-output.py:63 | FunctionDef | _materialize_raw
component/api-output.py:71 | FunctionDef | _same_filename
component/api-output.py:78 | FunctionDef | _row_belongs_to_doc
component/api-output.py:90 | FunctionDef | _filter_rows_for_doc
component/api-output.py:96 | FunctionDef | _pick_dynamic_template
component/api-output.py:110 | FunctionDef | _merge_template
component/api-output.py:150 | FunctionDef | _callback_headers
component/api-output.py:161 | FunctionDef | _deliver_callback
component/api-output.py:206 | FunctionDef | _stored_input_files
component/api-output.py:219 | FunctionDef | _normalize_for_api
component/api-output.py:261 | FunctionDef | _component_trace_context_values
component/api-output.py:269 | FunctionDef | _extract_doc_value_from_any
component/api-output.py:288 | FunctionDef | _overlay_component_traces
component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py:30 | FunctionDef | _is_punct_like_token
component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py:47 | FunctionDef | _normalize_token_fields
component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py:60 | FunctionDef | detect_lang
component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py:97 | FunctionDef | get_previous_cell_input
component/atrribution-gramatical/atripusion-gramatical-en-utilisant-les3ficherla.py:108 | FunctionDef | iter_sentences_from_input
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:86 | FunctionDef | _zeros_vec
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:90 | FunctionDef | _zeros_matrix
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:94 | FunctionDef | _vector_norm
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:98 | FunctionDef | _normalize_vector
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:105 | FunctionDef | _average_vectors
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:117 | FunctionDef | _env_true
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:122 | FunctionDef | _is_offline_mode
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:132 | FunctionDef | _resolve_model_for_loading
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:177 | FunctionDef | _is_punct_like_token
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:195 | FunctionDef | _normalize_token_fields
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:209 | FunctionDef | _strip_accents
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:214 | FunctionDef | _norm_token
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:220 | FunctionDef | detect_lang
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:234 | FunctionDef | get_previous_cell_input
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:246 | FunctionDef | iter_sentences_from_input
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:327 | FunctionDef | _basic_tokenize
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:331 | FunctionDef | _normalize_lemma
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:342 | FunctionDef | _guess_pos
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:386 | FunctionDef | _cosine
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:399 | FunctionDef | _build_pos_prototypes
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:432 | FunctionDef | _refine_pos_tags
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:519 | FunctionDef | _hash_vec
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:529 | ClassDef | _XLMRContextEncoder
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:530 | FunctionDef | __init__
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:577 | FunctionDef | encode_token_lists
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:639 | FunctionDef | _heuristic_ner
component/atrribution-gramatical/attribution-gramatical-100ml-xlmr.py:657 | FunctionDef | _run
component/atrribution-gramatical/engcode.py:12 | FunctionDef | print_install_help
component/atrribution-gramatical/engcode.py:39 | FunctionDef | _ensure_nltk
component/atrribution-gramatical/engcode.py:88 | ClassDef | Tok
component/atrribution-gramatical/engcode.py:103 | FunctionDef | _is_punct
component/atrribution-gramatical/engcode.py:106 | FunctionDef | _is_number
component/atrribution-gramatical/engcode.py:149 | FunctionDef | _lower
component/atrribution-gramatical/engcode.py:152 | FunctionDef | _has_apostrophe
component/atrribution-gramatical/engcode.py:155 | FunctionDef | _norm_apos
component/atrribution-gramatical/engcode.py:177 | FunctionDef | _looks_like_word
component/atrribution-gramatical/engcode.py:180 | FunctionDef | _split_plural_possessive
component/atrribution-gramatical/engcode.py:192 | FunctionDef | split_contractions
component/atrribution-gramatical/engcode.py:265 | FunctionDef | tokenize_with_spans
component/atrribution-gramatical/engcode.py:290 | FunctionDef | guess_pos_heur
component/atrribution-gramatical/engcode.py:330 | FunctionDef | map_penn_to_simple
component/atrribution-gramatical/engcode.py:367 | FunctionDef | pos_tag_simple
component/atrribution-gramatical/engcode.py:387 | FunctionDef | looks_like_nounish
component/atrribution-gramatical/engcode.py:400 | FunctionDef | resolve_s_contractions
component/atrribution-gramatical/engcode.py:435 | FunctionDef | _penn_to_wordnet_simple
component/atrribution-gramatical/engcode.py:447 | FunctionDef | lemma_en
component/atrribution-gramatical/engcode.py:488 | FunctionDef | fix_buffalo_pos
component/atrribution-gramatical/engcode.py:504 | FunctionDef | date_spans
component/atrribution-gramatical/engcode.py:529 | FunctionDef | currency_spans
component/atrribution-gramatical/engcode.py:547 | FunctionDef | load_hf_ner
component/atrribution-gramatical/engcode.py:569 | FunctionDef | _norm_ner_label
component/atrribution-gramatical/engcode.py:578 | FunctionDef | hf_spans
component/atrribution-gramatical/engcode.py:599 | FunctionDef | apply_spans_to_tokens
component/atrribution-gramatical/engcode.py:610 | FunctionDef | enforce_bio
component/atrribution-gramatical/engcode.py:619 | FunctionDef | fix_buffalo_ner
component/atrribution-gramatical/engcode.py:659 | FunctionDef | _is_initial_chain
component/atrribution-gramatical/engcode.py:665 | FunctionDef | drop_false_pers
component/atrribution-gramatical/engcode.py:691 | FunctionDef | _join_entity_tokens
component/atrribution-gramatical/engcode.py:698 | FunctionDef | entities_from_bio
component/atrribution-gramatical/engcode.py:721 | FunctionDef | print_table
component/atrribution-gramatical/engcode.py:730 | FunctionDef | fallback_ner_labels
component/atrribution-gramatical/engcode.py:756 | FunctionDef | run_one
component/atrribution-gramatical/engcode.py:859 | FunctionDef | split_input_into_sentences
component/atrribution-gramatical/engcode.py:873 | FunctionDef | get_ner_pipe
component/atrribution-gramatical/engcode.py:890 | FunctionDef | run_one_auto
component/atrribution-gramatical/engcode.py:894 | FunctionDef | run_from_previous_cell
component/atrribution-gramatical/engcode.py:951 | FunctionDef | main
component/atrribution-gramatical/frcode.py:12 | FunctionDef | print_install_help
component/atrribution-gramatical/frcode.py:99 | ClassDef | Tok
component/atrribution-gramatical/frcode.py:104 | FunctionDef | _norm_apo
component/atrribution-gramatical/frcode.py:107 | FunctionDef | _has_letters
component/atrribution-gramatical/frcode.py:110 | FunctionDef | _is_punct
component/atrribution-gramatical/frcode.py:113 | FunctionDef | _is_dash
component/atrribution-gramatical/frcode.py:116 | FunctionDef | _is_number
component/atrribution-gramatical/frcode.py:119 | FunctionDef | _is_acronym
component/atrribution-gramatical/frcode.py:122 | FunctionDef | _is_capitalized_word
component/atrribution-gramatical/frcode.py:125 | FunctionDef | _is_hyphenated_word
component/atrribution-gramatical/frcode.py:128 | FunctionDef | _is_email
component/atrribution-gramatical/frcode.py:131 | FunctionDef | _is_url
component/atrribution-gramatical/frcode.py:135 | FunctionDef | _is_ipv4
component/atrribution-gramatical/frcode.py:138 | FunctionDef | _is_hash
component/atrribution-gramatical/frcode.py:141 | FunctionDef | _is_id
component/atrribution-gramatical/frcode.py:144 | FunctionDef | _split_trailing_punct_if_needed
component/atrribution-gramatical/frcode.py:174 | FunctionDef | tokenize_raw
component/atrribution-gramatical/frcode.py:184 | FunctionDef | split_elisions
component/atrribution-gramatical/frcode.py:208 | FunctionDef | tokenize_with_spans
component/atrribution-gramatical/frcode.py:283 | FunctionDef | guess_pos
component/atrribution-gramatical/frcode.py:368 | FunctionDef | lemma_fr
component/atrribution-gramatical/frcode.py:442 | FunctionDef | date_spans
component/atrribution-gramatical/frcode.py:490 | FunctionDef | currency_spans
component/atrribution-gramatical/frcode.py:508 | FunctionDef | _norm_ner_label
component/atrribution-gramatical/frcode.py:517 | FunctionDef | load_hf_ner
component/atrribution-gramatical/frcode.py:539 | FunctionDef | hf_spans
component/atrribution-gramatical/frcode.py:560 | FunctionDef | apply_spans_to_tokens
component/atrribution-gramatical/frcode.py:571 | FunctionDef | enforce_bio
component/atrribution-gramatical/frcode.py:601 | FunctionDef | _is_initial_chain
component/atrribution-gramatical/frcode.py:607 | FunctionDef | drop_false_pers
component/atrribution-gramatical/frcode.py:630 | FunctionDef | ner_rules_spans
component/atrribution-gramatical/frcode.py:700 | FunctionDef | improve_pos_with_ner
component/atrribution-gramatical/frcode.py:713 | FunctionDef | _tok_lower
component/atrribution-gramatical/frcode.py:716 | FunctionDef | _is_function_word
component/atrribution-gramatical/frcode.py:720 | FunctionDef | _looks_like_proper
component/atrribution-gramatical/frcode.py:777 | FunctionDef | join_fr
component/atrribution-gramatical/frcode.py:794 | FunctionDef | entities_from_bio
component/atrribution-gramatical/frcode.py:816 | FunctionDef | print_table
component/atrribution-gramatical/frcode.py:828 | FunctionDef | run_one
component/atrribution-gramatical/frcode.py:914 | FunctionDef | split_input_into_sentences
component/atrribution-gramatical/frcode.py:928 | FunctionDef | get_ner_pipe
component/atrribution-gramatical/frcode.py:945 | FunctionDef | run_one_auto
component/atrribution-gramatical/frcode.py:948 | FunctionDef | run_from_previous_cell
component/atrribution-gramatical/frcode.py:1002 | FunctionDef | main
component/clasification.py:19 | FunctionDef | _load_json
component/clasification.py:25 | FunctionDef | _strip_accents
component/clasification.py:31 | FunctionDef | _norm_text
component/clasification.py:36 | FunctionDef | _norm_keyword
component/clasification.py:40 | FunctionDef | _ensure_kw_dict
component/clasification.py:59 | FunctionDef | load_classification_configs
component/clasification.py:92 | FunctionDef | _get_previous_cell_input
component/clasification.py:105 | FunctionDef | _build_DOCS_from_input
component/clasification.py:161 | FunctionDef | _doc_text_len
component/clasification.py:169 | FunctionDef | _drop_empty_duplicates
component/clasification.py:185 | FunctionDef | classify_scores
component/clasification.py:197 | FunctionDef | _push_audit
component/clasification.py:213 | FunctionDef | add_score
component/clasification.py:281 | FunctionDef | decide
component/clasification.py:339 | FunctionDef | _bucket_pairs
component/clasification.py:400 | FunctionDef | _fmt
component/detection-signature-chachet-codebarr.py:28 | FunctionDef | resolve_runtime_input_path
component/detection-signature-chachet-codebarr.py:42 | FunctionDef | _safe_list
component/detection-signature-chachet-codebarr.py:46 | FunctionDef | _safe_int
component/detection-signature-chachet-codebarr.py:53 | FunctionDef | _safe_float
component/detection-signature-chachet-codebarr.py:60 | FunctionDef | _doc_key
component/detection-signature-chachet-codebarr.py:70 | FunctionDef | _build_source_path_map
component/detection-signature-chachet-codebarr.py:74 | FunctionDef | _add
component/detection-signature-chachet-codebarr.py:96 | FunctionDef | _iter_docs
component/detection-signature-chachet-codebarr.py:103 | FunctionDef | _resolve_source_path
component/detection-signature-chachet-codebarr.py:118 | FunctionDef | _pil_to_cv_bgr
component/detection-signature-chachet-codebarr.py:125 | FunctionDef | _open_image_frames
component/detection-signature-chachet-codebarr.py:136 | FunctionDef | _pdf_page_count
component/detection-signature-chachet-codebarr.py:154 | FunctionDef | _sample_page_numbers
component/detection-signature-chachet-codebarr.py:185 | FunctionDef | _render_pdf_page
component/detection-signature-chachet-codebarr.py:221 | FunctionDef | _render_pdf_pages
component/detection-signature-chachet-codebarr.py:235 | FunctionDef | _load_doc_pages
component/detection-signature-chachet-codebarr.py:248 | FunctionDef | _prepare_arrays
component/detection-signature-chachet-codebarr.py:269 | FunctionDef | _transition_density
component/detection-signature-chachet-codebarr.py:277 | FunctionDef | _active_segments
component/detection-signature-chachet-codebarr.py:313 | FunctionDef | _mask_components
component/detection-signature-chachet-codebarr.py:351 | FunctionDef | _border_center_ratio
component/detection-signature-chachet-codebarr.py:368 | FunctionDef | _score_signature
component/detection-signature-chachet-codebarr.py:444 | FunctionDef | _score_stamp
component/detection-signature-chachet-codebarr.py:479 | FunctionDef | _score_qrcode
component/detection-signature-chachet-codebarr.py:510 | FunctionDef | _score_barcode
component/detection-signature-chachet-codebarr.py:544 | FunctionDef | _detect_qr_barcode_decoders
component/detection-signature-chachet-codebarr.py:627 | FunctionDef | _bbox_iou
component/detection-signature-chachet-codebarr.py:647 | FunctionDef | _scan_windows
component/detection-signature-chachet-codebarr.py:754 | FunctionDef | _detect_page_marks
component/detection-signature-chachet-codebarr.py:768 | FunctionDef | _dedupe_doc_detections
component/detection-signature-chachet-codebarr.py:795 | FunctionDef | run
component/elasticsearch.py:22 | FunctionDef | _normalize_ids
component/elasticsearch.py:32 | FunctionDef | run
component/extraction/extraction-regles-100ml.py:15 | FunctionDef | _tokenize
component/extraction/extraction-regles-100ml.py:19 | FunctionDef | _safe_int
component/extraction/extraction-regles-100ml.py:26 | FunctionDef | _doc_key
component/extraction/extraction-regles-100ml.py:34 | FunctionDef | _iter_doc_chunks
component/extraction/extraction-regles-100ml.py:77 | FunctionDef | _build_docs_lookup
component/extraction/extraction-regles-100ml.py:96 | FunctionDef | _collect_query_terms
component/extraction/extraction-regles-100ml.py:125 | FunctionDef | _bm25_scores
component/extraction/extraction-regles-100ml.py:166 | FunctionDef | _run_base_extraction
component/extraction/extraction-regles-100ml.py:172 | FunctionDef | _add_bm25
component/extraction/extraction-regles-50ml.py:15 | FunctionDef | _tokenize
component/extraction/extraction-regles-50ml.py:19 | FunctionDef | _safe_int
component/extraction/extraction-regles-50ml.py:26 | FunctionDef | _doc_key
component/extraction/extraction-regles-50ml.py:34 | FunctionDef | _iter_doc_chunks
component/extraction/extraction-regles-50ml.py:77 | FunctionDef | _build_docs_lookup
component/extraction/extraction-regles-50ml.py:96 | FunctionDef | _collect_query_terms
component/extraction/extraction-regles-50ml.py:125 | FunctionDef | _bm25_scores
component/extraction/extraction-regles-50ml.py:166 | FunctionDef | _run_base_extraction
component/extraction/extraction-regles-50ml.py:172 | FunctionDef | _add_bm25
component/extraction/extraction-regles-yaml.py:115 | FunctionDef | _load_yaml
component/extraction/extraction-regles-yaml.py:125 | FunctionDef | _load_json
component/extraction/extraction-regles-yaml.py:133 | FunctionDef | _load_json_like_yaml
component/extraction/extraction-regles-yaml.py:138 | FunctionDef | _load_routes
component/extraction/extraction-regles-yaml.py:158 | FunctionDef | _resolve_rulesets
component/extraction/extraction-regles-yaml.py:173 | FunctionDef | _normalize_doc_type
component/extraction/extraction-regles-yaml.py:180 | FunctionDef | _is_common_fallback_doc_type
component/extraction/extraction-regles-yaml.py:184 | FunctionDef | load_extractors_for
component/extraction/extraction-regles-yaml.py:220 | FunctionDef | _get_input_docs
component/extraction/extraction-regles-yaml.py:229 | FunctionDef | _page_text_from_page
component/extraction/extraction-regles-yaml.py:245 | FunctionDef | _norm_text
component/extraction/extraction-regles-yaml.py:252 | FunctionDef | _normalize_digits
component/extraction/extraction-regles-yaml.py:256 | FunctionDef | _strip_control_and_escapes
component/extraction/extraction-regles-yaml.py:272 | FunctionDef | _clean_value
component/extraction/extraction-regles-yaml.py:279 | FunctionDef | _normalize_email_value
component/extraction/extraction-regles-yaml.py:289 | FunctionDef | _normalize_url_value
component/extraction/extraction-regles-yaml.py:298 | FunctionDef | _normalize_phone_value
component/extraction/extraction-regles-yaml.py:309 | FunctionDef | _first_non_empty_line
component/extraction/extraction-regles-yaml.py:318 | FunctionDef | _find_label_in_line
component/extraction/extraction-regles-yaml.py:338 | FunctionDef | _value_after_label
component/extraction/extraction-regles-yaml.py:363 | FunctionDef | _split_tokens_rough
component/extraction/extraction-regles-yaml.py:383 | FunctionDef | _extract_date_value
component/extraction/extraction-regles-yaml.py:405 | FunctionDef | _extract_amount_value
component/extraction/extraction-regles-yaml.py:418 | FunctionDef | _extract_currency_value
component/extraction/extraction-regles-yaml.py:426 | FunctionDef | _detect_emails
component/extraction/extraction-regles-yaml.py:440 | FunctionDef | _detect_urls
component/extraction/extraction-regles-yaml.py:450 | FunctionDef | _detect_phones
component/extraction/extraction-regles-yaml.py:468 | FunctionDef | _detect_values_by_type
component/extraction/extraction-regles-yaml.py:488 | FunctionDef | _normalize_value_by_type
component/extraction/extraction-regles-yaml.py:508 | FunctionDef | _compile_regex
component/extraction/extraction-regles-yaml.py:523 | FunctionDef | _make_match
component/extraction/extraction-regles-yaml.py:540 | FunctionDef | _apply_extractor_to_page
component/extraction/extraction-regles-yaml.py:557 | FunctionDef | _record
component/extraction/extraction-regles-yaml.py:647 | FunctionDef | _build_cls_map
component/extraction/extraction-regles-yaml.py:658 | FunctionDef | _classification_for
component/extraction/extraction-regles-yaml.py:668 | FunctionDef | _doc_text_score
component/extraction/extraction-regles-yaml.py:681 | FunctionDef | _dedupe_docs
component/extraction/extraction-regles-yaml.py:709 | FunctionDef | _unique_keep_order
component/extraction/extraction-regles-yaml.py:720 | FunctionDef | run
component/extraction/extraction-regles.py:25 | FunctionDef | _load_json
component/extraction/extraction-regles.py:32 | FunctionDef | _compile_regex
component/extraction/extraction-regles.py:48 | FunctionDef | _load_routes
component/extraction/extraction-regles.py:65 | FunctionDef | _resolve_rulesets
component/extraction/extraction-regles.py:80 | FunctionDef | _normalize_doc_type
component/extraction/extraction-regles.py:87 | FunctionDef | _is_common_fallback_doc_type
component/extraction/extraction-regles.py:91 | FunctionDef | load_extractors_for
component/extraction/extraction-regles.py:122 | FunctionDef | _get_input_docs
component/extraction/extraction-regles.py:133 | FunctionDef | _page_text_from_page
component/extraction/extraction-regles.py:150 | FunctionDef | apply_extractors_to_page
component/extraction/extraction-regles.py:165 | FunctionDef | _record
component/extraction/extraction-regles.py:199 | FunctionDef | _build_cls_map
component/extraction/extraction-regles.py:210 | FunctionDef | _classification_for
component/extraction/extraction-regles.py:220 | FunctionDef | _doc_text_score
component/extraction/extraction-regles.py:233 | FunctionDef | _dedupe_docs
component/extraction/extraction-regles.py:261 | FunctionDef | run
component/fusion_resultats.py:34 | FunctionDef | ns
component/fusion_resultats.py:38 | FunctionDef | first
component/fusion_resultats.py:42 | FunctionDef | merge_list
component/fusion_resultats.py:50 | FunctionDef | _safe_list
component/fusion_resultats.py:54 | FunctionDef | _safe_dict
component/fusion_resultats.py:58 | FunctionDef | _safe_load_json
component/fusion_resultats.py:67 | FunctionDef | _safe_int
component/fusion_resultats.py:74 | FunctionDef | _interdoc_aliases
component/fusion_resultats.py:96 | FunctionDef | _interdoc_link_ids_for_doc
component/fusion_resultats.py:115 | FunctionDef | _interdoc_output
component/fusion_resultats.py:144 | FunctionDef | _iso_now
component/fusion_resultats.py:148 | FunctionDef | _basename
component/fusion_resultats.py:157 | FunctionDef | _norm_filename
component/fusion_resultats.py:161 | FunctionDef | _same_filename
component/fusion_resultats.py:167 | FunctionDef | _safe_non_negative_int
component/fusion_resultats.py:177 | FunctionDef | _size_from_paths
component/fusion_resultats.py:201 | FunctionDef | _size_maps_from_pretraitement
component/fusion_resultats.py:232 | FunctionDef | _resolve_file_size
component/fusion_resultats.py:257 | FunctionDef | _row_belongs_to_doc
component/fusion_resultats.py:269 | FunctionDef | _filter_rows_for_doc
component/fusion_resultats.py:279 | FunctionDef | _component_traces
component/fusion_resultats.py:283 | FunctionDef | _extract_doc_value_from_any
component/fusion_resultats.py:302 | FunctionDef | _component_trace_context_values
component/fusion_resultats.py:310 | FunctionDef | _component_trace_doc_view
component/fusion_resultats.py:340 | FunctionDef | _pipeline_component_runs
component/fusion_resultats.py:361 | FunctionDef | _auto_component_views
component/fusion_resultats.py:384 | FunctionDef | _doc_key
component/fusion_resultats.py:397 | FunctionDef | _doc_text_score
component/fusion_resultats.py:414 | FunctionDef | _dedupe_docs
component/fusion_resultats.py:454 | FunctionDef | _normalize_pages_from_doc
component/fusion_resultats.py:468 | FunctionDef | _derive_doc_page_count
component/fusion_resultats.py:495 | FunctionDef | _filter_and_dedupe_extractions
component/fusion_resultats.py:536 | FunctionDef | _default_nlp_tokens_index
component/fusion_resultats.py:544 | FunctionDef | _search_index
component/fusion_resultats.py:564 | FunctionDef | _fetch_nlp_tokens
component/fusion_resultats.py:641 | FunctionDef | _structure_nlp_tokens
component/fusion_resultats.py:689 | FunctionDef | _nlp_from_es_and_ctx
component/fusion_resultats.py:879 | FunctionDef | _build_map
component/fusion_resultats.py:894 | FunctionDef | _pick_from_map
component/fusion_resultats.py:906 | FunctionDef | _same_doc_hint
component/fusion_resultats.py:915 | FunctionDef | _extract_component_views
component/fusion_resultats.py:1122 | FunctionDef | _to_document_output
component/fusion_resultats.py:1173 | FunctionDef | _final_output
component/fusion_resultats.py:1205 | FunctionDef | extract_text_raw
component/fusion_resultats.py:1234 | FunctionDef | extract_pages_meta
component/fusion_resultats.py:1240 | FunctionDef | extract_pages
component/fusion_resultats.py:1267 | FunctionDef | extract_classification
component/fusion_resultats.py:1273 | FunctionDef | extract_doc_type
component/fusion_resultats.py:1277 | FunctionDef | extract_extractions
component/fusion_resultats.py:1281 | FunctionDef | extract_detected_languages
component/fusion_resultats.py:1285 | FunctionDef | _split_lang_values
component/fusion_resultats.py:1300 | FunctionDef | _add_lang_counts
component/fusion_resultats.py:1331 | FunctionDef | _derive_doc_languages
component/fusion_resultats.py:1380 | FunctionDef | _text_from_tok_pages
component/fusion_resultats.py:1391 | FunctionDef | _pages_from_final_doc
component/fusion_resultats.py:1405 | FunctionDef | _collect_local_tokens_for_doc
component/fusion_resultats.py:1475 | FunctionDef | _pick_local_final_doc
component/fusion_resultats.py:1488 | FunctionDef | _pick_local_classification
component/fusion_resultats.py:1501 | FunctionDef | _build_local_payload_for_doc
component/fusion_resultats.py:1633 | FunctionDef | build_payloads_from_context
component/fusion_resultats.py:1666 | FunctionDef | build_payload_from_context
component/fusion_resultats.py:1863 | FunctionDef | _es_text_from_pages
component/fusion_resultats.py:1874 | FunctionDef | _es_pages
component/fusion_resultats.py:1890 | FunctionDef | build_payload_from_es_source
component/fusion_resultats.py:2011 | FunctionDef | build_payloads_from_es
component/fusion_resultats.py:2061 | FunctionDef | _normalize_pipeline_profile
component/fusion_resultats.py:2070 | FunctionDef | _active_doc_section
component/fusion_resultats.py:2079 | FunctionDef | _purge_inactive_profile_payload
component/fusion_resultats.py:2136 | FunctionDef | _profile_norm_term
component/fusion_resultats.py:2146 | FunctionDef | _profile_doc_key
component/fusion_resultats.py:2156 | FunctionDef | _profile_doc_aliases
component/fusion_resultats.py:2168 | FunctionDef | _profile_index_rows
component/fusion_resultats.py:2184 | FunctionDef | _profile_group_rows
component/fusion_resultats.py:2201 | FunctionDef | _profile_pick_bm25
component/fusion_resultats.py:2209 | FunctionDef | _is_grammar_noise
component/fusion_resultats.py:2225 | FunctionDef | _build_grammar_block_map
component/fusion_resultats.py:2244 | FunctionDef | _collect_blocked_terms
component/fusion_resultats.py:2257 | FunctionDef | _is_blocked_topic_term
component/fusion_resultats.py:2273 | FunctionDef | _filter_topics
component/fusion_resultats.py:2299 | FunctionDef | _filter_chunk_topics
component/fusion_resultats.py:2317 | FunctionDef | _augment_payload_for_profile
component/fusion_resultats.py:2476 | FunctionDef | _augment_payload_with_default_tables
component/fusion_resultats.py:2560 | FunctionDef | _append_totals_quality_check
component/fusion_resultats.py:2594 | FunctionDef | _augment_payload_with_totals_verification
component/fusion_resultats.py:2731 | FunctionDef | _augment_payload_with_visual_marks_100ml
component/fusion_resultats.py:2851 | FunctionDef | main
component/liaison-inter-docs.py:88 | FunctionDef | _iso_now
component/liaison-inter-docs.py:92 | FunctionDef | _safe_list
component/liaison-inter-docs.py:96 | FunctionDef | _safe_int
component/liaison-inter-docs.py:103 | FunctionDef | _safe_float
component/liaison-inter-docs.py:110 | FunctionDef | _normalize_term
component/liaison-inter-docs.py:120 | FunctionDef | _normalize_pipeline_profile
component/liaison-inter-docs.py:129 | FunctionDef | _active_topic_sources
component/liaison-inter-docs.py:138 | FunctionDef | _filename_aliases
component/liaison-inter-docs.py:154 | FunctionDef | _sentence_key
component/liaison-inter-docs.py:158 | FunctionDef | _is_semantic_pos
component/liaison-inter-docs.py:174 | FunctionDef | _is_informative_term
component/liaison-inter-docs.py:191 | FunctionDef | _tokenize_terms
component/liaison-inter-docs.py:201 | FunctionDef | _split_to_sentences
component/liaison-inter-docs.py:211 | FunctionDef | _clip_text
component/liaison-inter-docs.py:218 | FunctionDef | _coerce_vector
component/liaison-inter-docs.py:230 | FunctionDef | _vector_norm
component/liaison-inter-docs.py:234 | FunctionDef | _unit_vector
component/liaison-inter-docs.py:244 | FunctionDef | _unit_cosine_similarity
component/liaison-inter-docs.py:255 | FunctionDef | _cosine_similarity
component/liaison-inter-docs.py:273 | FunctionDef | _mean_vector
component/liaison-inter-docs.py:291 | FunctionDef | _doc_key
component/liaison-inter-docs.py:301 | FunctionDef | _doc_aliases
component/liaison-inter-docs.py:315 | FunctionDef | _extract_classification_terms
component/liaison-inter-docs.py:334 | FunctionDef | _build_classification_index
component/liaison-inter-docs.py:348 | FunctionDef | _build_semantic_sentence_index
component/liaison-inter-docs.py:377 | FunctionDef | _lookup_semantic_terms
component/liaison-inter-docs.py:391 | FunctionDef | _iter_doc_sentences
component/liaison-inter-docs.py:444 | FunctionDef | _doc_text_score
component/liaison-inter-docs.py:452 | FunctionDef | _dedupe_docs
component/liaison-inter-docs.py:470 | FunctionDef | _index_topic_rows
component/liaison-inter-docs.py:482 | FunctionDef | _vector_profile
component/liaison-inter-docs.py:507 | FunctionDef | _build_vector_index
component/liaison-inter-docs.py:551 | FunctionDef | _extract_topics
component/liaison-inter-docs.py:578 | FunctionDef | _chunk_topic_terms
component/liaison-inter-docs.py:592 | FunctionDef | _prepare_docs
component/liaison-inter-docs.py:712 | FunctionDef | _score_informative_term
component/liaison-inter-docs.py:727 | FunctionDef | _sentence_match_score
component/liaison-inter-docs.py:780 | FunctionDef | _topic_examples_for_doc
component/liaison-inter-docs.py:803 | FunctionDef | _sorted_shared_terms
component/liaison-inter-docs.py:809 | FunctionDef | _chunk_candidate_priority
component/liaison-inter-docs.py:827 | FunctionDef | _select_chunk_candidates
component/liaison-inter-docs.py:847 | FunctionDef | _vector_chunk_matches
component/liaison-inter-docs.py:978 | FunctionDef | _build_link
component/liaison-inter-docs.py:1158 | FunctionDef | _compute_links
component/liaison-inter-docs.py:1179 | FunctionDef | _build_doc_links_index
component/liaison-inter-docs.py:1200 | FunctionDef | run
component/output-txt.py:30 | FunctionDef | resolve_runtime_input_path
component/output-txt.py:33 | FunctionDef | parse_git_lfs_pointer_path
component/output-txt.py:64 | FunctionDef | build_config
component/output-txt.py:69 | ClassDef | _DummyFT
component/output-txt.py:70 | FunctionDef | __init__
component/output-txt.py:73 | FunctionDef | detect_path_type
component/output-txt.py:85 | FunctionDef | _median
component/output-txt.py:96 | FunctionDef | _estimate_char_metrics
component/output-txt.py:140 | FunctionDef | _render_layout_from_data
component/output-txt.py:251 | FunctionDef | _basename
component/output-txt.py:261 | FunctionDef | _safe_file_size
component/output-txt.py:270 | FunctionDef | _doc_size_from_sources
component/output-txt.py:384 | FunctionDef | _get_pdf_reader_with_name
component/output-txt.py:396 | FunctionDef | _pdf_extract_text_preserve_layout
component/output-txt.py:408 | FunctionDef | _docx_xml_to_text
component/output-txt.py:412 | FunctionDef | _docx_xml_to_pages
component/output-txt.py:420 | FunctionDef | _flush_page_break
component/output-txt.py:450 | FunctionDef | _docx_app_page_count
component/output-txt.py:464 | FunctionDef | _pdf_page_count
component/output-txt.py:483 | FunctionDef | _pad_pages_text
component/output-txt.py:498 | FunctionDef | _pptx_slide_xml_to_text
component/output-txt.py:518 | FunctionDef | _xlsx_col_to_index
component/output-txt.py:526 | FunctionDef | _xlsx_shared_strings
component/output-txt.py:538 | FunctionDef | _xlsx_sheet_to_text
component/output-txt.py:588 | FunctionDef | _odf_content_to_text
component/output-txt.py:592 | FunctionDef | walk
component/output-txt.py:619 | FunctionDef | _html_bytes_to_text_preserve
component/output-txt.py:630 | FunctionDef | extract_text_native
component/pretraitement-de-docs.py:15 | FunctionDef | resolve_runtime_input_path
component/pretraitement-de-docs.py:43 | ClassDef | FileType
component/pretraitement-de-docs.py:51 | FunctionDef | normalize_input_files
component/pretraitement-de-docs.py:66 | FunctionDef | resolve_path
component/pretraitement-de-docs.py:86 | FunctionDef | _safe_file_size
component/pretraitement-de-docs.py:95 | FunctionDef | _read_head
component/pretraitement-de-docs.py:101 | FunctionDef | detect_path_type
component/pretraitement-de-docs.py:172 | FunctionDef | _xml_text_len
component/pretraitement-de-docs.py:186 | FunctionDef | _zip_has_text
component/pretraitement-de-docs.py:257 | FunctionDef | _get_pdf_reader
component/pretraitement-de-docs.py:270 | FunctionDef | _pdf_has_text
component/pretraitement-de-docs.py:334 | FunctionDef | content_kind_two_states
component/pretraitement-de-docs.py:363 | FunctionDef | analyze_many_two_states
component/si-image-pretraiter-sinonpass-le-doc.py:24 | FunctionDef | resolve_runtime_input_path
component/si-image-pretraiter-sinonpass-le-doc.py:27 | FunctionDef | parse_git_lfs_pointer_path
component/si-image-pretraiter-sinonpass-le-doc.py:58 | ClassDef | FileType
component/si-image-pretraiter-sinonpass-le-doc.py:64 | FunctionDef | _read_head
component/si-image-pretraiter-sinonpass-le-doc.py:70 | FunctionDef | normalize_input_files
component/si-image-pretraiter-sinonpass-le-doc.py:84 | FunctionDef | resolve_path
component/si-image-pretraiter-sinonpass-le-doc.py:98 | FunctionDef | detect_path_type
component/si-image-pretraiter-sinonpass-le-doc.py:161 | FunctionDef | _xml_text_len
component/si-image-pretraiter-sinonpass-le-doc.py:174 | FunctionDef | _zip_has_text
component/si-image-pretraiter-sinonpass-le-doc.py:237 | FunctionDef | _get_pdf_reader
component/si-image-pretraiter-sinonpass-le-doc.py:249 | FunctionDef | _pdf_has_text
component/si-image-pretraiter-sinonpass-le-doc.py:279 | FunctionDef | content_kind_two_states
component/si-image-pretraiter-sinonpass-le-doc.py:352 | ClassDef | EnhanceOptions
component/si-image-pretraiter-sinonpass-le-doc.py:370 | FunctionDef | build_config
component/si-image-pretraiter-sinonpass-le-doc.py:396 | FunctionDef | ensure_environment
component/si-image-pretraiter-sinonpass-le-doc.py:416 | FunctionDef | auto_rotate_if_needed
component/si-image-pretraiter-sinonpass-le-doc.py:436 | FunctionDef | preprocess_image
component/si-image-pretraiter-sinonpass-le-doc.py:517 | FunctionDef | parse_args
component/si-image-pretraiter-sinonpass-le-doc.py:590 | FunctionDef | _normalize_input_files
component/si-image-pretraiter-sinonpass-le-doc.py:622 | FunctionDef | _load_images_from_path
component/table_extraction/table-extraction.py:14 | FunctionDef | _resolve_profile
component/table_extraction/table_extraction_lib.py:12 | FunctionDef | resolve_runtime_input_path
component/table_extraction/table_extraction_lib.py:246 | FunctionDef | _safe_list
component/table_extraction/table_extraction_lib.py:250 | FunctionDef | _doc_key
component/table_extraction/table_extraction_lib.py:260 | FunctionDef | _doc_text_score
component/table_extraction/table_extraction_lib.py:270 | FunctionDef | _dedupe_docs
component/table_extraction/table_extraction_lib.py:286 | FunctionDef | _norm_text
component/table_extraction/table_extraction_lib.py:296 | FunctionDef | _compact_spaces
component/table_extraction/table_extraction_lib.py:302 | FunctionDef | _line_token_spans
component/table_extraction/table_extraction_lib.py:313 | FunctionDef | _line_segment_spans
component/table_extraction/table_extraction_lib.py:343 | FunctionDef | _line_geometry
component/table_extraction/table_extraction_lib.py:394 | FunctionDef | _alignment_with_neighbor
component/table_extraction/table_extraction_lib.py:406 | FunctionDef | _line_tabularity_score
component/table_extraction/table_extraction_lib.py:426 | FunctionDef | _anchors_from_header_geom
component/table_extraction/table_extraction_lib.py:444 | FunctionDef | _anchor_matches
component/table_extraction/table_extraction_lib.py:457 | FunctionDef | _cells_from_anchors
component/table_extraction/table_extraction_lib.py:478 | FunctionDef | _collect_blocks_anchor
component/table_extraction/table_extraction_lib.py:580 | FunctionDef | _merge_blocks
component/table_extraction/table_extraction_lib.py:598 | FunctionDef | _split_line_cells
component/table_extraction/table_extraction_lib.py:626 | FunctionDef | _is_probable_code
component/table_extraction/table_extraction_lib.py:637 | FunctionDef | _normalize_reference_code
component/table_extraction/table_extraction_lib.py:651 | FunctionDef | _extract_reference_code
component/table_extraction/table_extraction_lib.py:667 | FunctionDef | _clean_ocr_label
component/table_extraction/table_extraction_lib.py:676 | FunctionDef | _guess_product_number_from_reference
component/table_extraction/table_extraction_lib.py:690 | FunctionDef | _token_is_numeric_tail
component/table_extraction/table_extraction_lib.py:705 | FunctionDef | _merge_numeric_tokens
component/table_extraction/table_extraction_lib.py:727 | FunctionDef | _split_dense_numeric_tail
component/table_extraction/table_extraction_lib.py:751 | FunctionDef | _split_dense_header_cells
component/table_extraction/table_extraction_lib.py:780 | FunctionDef | _is_numeric_like
component/table_extraction/table_extraction_lib.py:795 | FunctionDef | _to_amount
component/table_extraction/table_extraction_lib.py:836 | FunctionDef | _normalize_number_token
component/table_extraction/table_extraction_lib.py:878 | FunctionDef | _to_quantity
component/table_extraction/table_extraction_lib.py:903 | FunctionDef | _header_score
component/table_extraction/table_extraction_lib.py:930 | FunctionDef | _detect_header_map
component/table_extraction/table_extraction_lib.py:954 | FunctionDef | _count_norm_hits
component/table_extraction/table_extraction_lib.py:958 | FunctionDef | _line_header_hint_score
component/table_extraction/table_extraction_lib.py:965 | FunctionDef | _line_footer_hint_score
component/table_extraction/table_extraction_lib.py:970 | FunctionDef | _is_totals_or_footer_label
component/table_extraction/table_extraction_lib.py:981 | FunctionDef | _is_footer_like_line
component/table_extraction/table_extraction_lib.py:999 | FunctionDef | _is_header_like_line
component/table_extraction/table_extraction_lib.py:1011 | FunctionDef | _line_looks_tabular
component/table_extraction/table_extraction_lib.py:1043 | FunctionDef | _is_strong_single_row
component/table_extraction/table_extraction_lib.py:1056 | FunctionDef | _collect_blocks
component/table_extraction/table_extraction_lib.py:1095 | FunctionDef | _infer_map_from_rows
component/table_extraction/table_extraction_lib.py:1192 | FunctionDef | _extract_totals_from_line
component/table_extraction/table_extraction_lib.py:1225 | FunctionDef | _row_to_item
component/table_extraction/table_extraction_lib.py:1233 | FunctionDef | get
component/table_extraction/table_extraction_lib.py:1347 | FunctionDef | _is_complete_line_item
component/table_extraction/table_extraction_lib.py:1362 | FunctionDef | _filter_complete_line_items
component/table_extraction/table_extraction_lib.py:1418 | FunctionDef | _extract_doc_tables
component/table_extraction/table_extraction_lib.py:1599 | FunctionDef | _augment_code_only_rows_from_header
component/table_extraction/table_extraction_lib.py:1693 | FunctionDef | _augment_text_line_items_from_header
component/table_extraction/table_extraction_lib.py:1861 | FunctionDef | _build_source_path_map
component/table_extraction/table_extraction_lib.py:1865 | FunctionDef | _add
component/table_extraction/table_extraction_lib.py:1887 | FunctionDef | _should_run_ocr_fallback
component/table_extraction/table_extraction_lib.py:1919 | FunctionDef | _parse_ocr_row_line
component/table_extraction/table_extraction_lib.py:2004 | FunctionDef | _parse_ocr_table_rows
component/table_extraction/table_extraction_lib.py:2061 | FunctionDef | _guess_total_field_key
component/table_extraction/table_extraction_lib.py:2069 | FunctionDef | _parse_ocr_totals_rows
component/table_extraction/table_extraction_lib.py:2125 | FunctionDef | _ocr_fallback_tables_from_image
component/table_extraction/table_extraction_lib.py:2142 | FunctionDef | _build_variants
component/table_extraction/table_extraction_lib.py:2211 | FunctionDef | _merge_ocr_fallback_rows
component/table_extraction/table_extraction_lib.py:2224 | FunctionDef | _sig
component/table_extraction/table_extraction_lib.py:2316 | FunctionDef | _merge_ocr_totals_rows
component/table_extraction/table_extraction_lib.py:2351 | FunctionDef | _extract_code_keys_from_text
component/table_extraction/table_extraction_lib.py:2364 | FunctionDef | _table_metrics
component/table_extraction/table_extraction_lib.py:2386 | FunctionDef | _infer_detected_columns_from_rows
component/table_extraction/table_extraction_lib.py:2404 | FunctionDef | _prune_redundant_tables
component/table_extraction/table_extraction_lib.py:2482 | FunctionDef | run_table_extraction
component/tokenisation_layout/tokenisation-layout-100ml.py:35 | FunctionDef | _safe_int
component/tokenisation_layout/tokenisation-layout-100ml.py:42 | FunctionDef | _norm_token
component/tokenisation_layout/tokenisation-layout-100ml.py:46 | FunctionDef | _tokenize
component/tokenisation_layout/tokenisation-layout-100ml.py:50 | FunctionDef | _norm_topic_term
component/tokenisation_layout/tokenisation-layout-100ml.py:60 | FunctionDef | _is_topic_candidate
component/tokenisation_layout/tokenisation-layout-100ml.py:81 | FunctionDef | _tokens_to_topic_terms
component/tokenisation_layout/tokenisation-layout-100ml.py:97 | FunctionDef | _score_term_quality
component/tokenisation_layout/tokenisation-layout-100ml.py:108 | FunctionDef | _prune_topic_redundancy
component/tokenisation_layout/tokenisation-layout-100ml.py:129 | FunctionDef | _extract_terms_from_keyword_item
component/tokenisation_layout/tokenisation-layout-100ml.py:141 | FunctionDef | _build_topic_boost_terms
component/tokenisation_layout/tokenisation-layout-100ml.py:163 | FunctionDef | _apply
component/tokenisation_layout/tokenisation-layout-100ml.py:188 | FunctionDef | _hash_index_sign
component/tokenisation_layout/tokenisation-layout-100ml.py:195 | FunctionDef | _vector_norm
component/tokenisation_layout/tokenisation-layout-100ml.py:199 | FunctionDef | _hash_text_vector
component/tokenisation_layout/tokenisation-layout-100ml.py:215 | FunctionDef | _mean_vectors
component/tokenisation_layout/tokenisation-layout-100ml.py:230 | FunctionDef | _to_list
component/tokenisation_layout/tokenisation-layout-100ml.py:234 | FunctionDef | _iter_doc_chunks
component/tokenisation_layout/tokenisation-layout-100ml.py:255 | FunctionDef | _extract_topics_from_chunks
component/tokenisation_layout/tokenisation-layout-100ml.py:290 | FunctionDef | _extract_chunk_topics
component/tokenisation_layout/tokenisation-layout-100ml.py:319 | FunctionDef | _doc_key
component/tokenisation_layout/tokenisation-layout-100ml.py:327 | ClassDef | _TransformerEmbedder
component/tokenisation_layout/tokenisation-layout-100ml.py:328 | FunctionDef | __init__
component/tokenisation_layout/tokenisation-layout-100ml.py:361 | FunctionDef | embed_texts
component/tokenisation_layout/tokenisation-layout-100ml.py:413 | FunctionDef | _run_base_tokenisation
component/tokenisation_layout/tokenisation-layout-100ml.py:419 | FunctionDef | _augment_with_ml100
component/tokenisation_layout/tokenisation-layout-50ml.py:35 | FunctionDef | _safe_int
component/tokenisation_layout/tokenisation-layout-50ml.py:42 | FunctionDef | _norm_token
component/tokenisation_layout/tokenisation-layout-50ml.py:46 | FunctionDef | _tokenize
component/tokenisation_layout/tokenisation-layout-50ml.py:50 | FunctionDef | _norm_topic_term
component/tokenisation_layout/tokenisation-layout-50ml.py:61 | FunctionDef | _is_topic_candidate
component/tokenisation_layout/tokenisation-layout-50ml.py:82 | FunctionDef | _tokens_to_topic_terms
component/tokenisation_layout/tokenisation-layout-50ml.py:100 | FunctionDef | _score_term_quality
component/tokenisation_layout/tokenisation-layout-50ml.py:112 | FunctionDef | _prune_topic_redundancy
component/tokenisation_layout/tokenisation-layout-50ml.py:134 | FunctionDef | _extract_terms_from_keyword_item
component/tokenisation_layout/tokenisation-layout-50ml.py:147 | FunctionDef | _build_topic_boost_terms
component/tokenisation_layout/tokenisation-layout-50ml.py:170 | FunctionDef | _apply
component/tokenisation_layout/tokenisation-layout-50ml.py:195 | FunctionDef | _char_ngrams
component/tokenisation_layout/tokenisation-layout-50ml.py:211 | FunctionDef | _hash_index_sign
component/tokenisation_layout/tokenisation-layout-50ml.py:218 | FunctionDef | _vector_norm
component/tokenisation_layout/tokenisation-layout-50ml.py:222 | FunctionDef | _fasttext_like_vector
component/tokenisation_layout/tokenisation-layout-50ml.py:234 | FunctionDef | _mean_vectors
component/tokenisation_layout/tokenisation-layout-50ml.py:249 | FunctionDef | _to_list
component/tokenisation_layout/tokenisation-layout-50ml.py:253 | FunctionDef | _iter_doc_chunks
component/tokenisation_layout/tokenisation-layout-50ml.py:274 | FunctionDef | _extract_topics_from_chunks
component/tokenisation_layout/tokenisation-layout-50ml.py:309 | FunctionDef | _extract_chunk_topics
component/tokenisation_layout/tokenisation-layout-50ml.py:338 | FunctionDef | _doc_key
component/tokenisation_layout/tokenisation-layout-50ml.py:346 | FunctionDef | _run_base_tokenisation
component/tokenisation_layout/tokenisation-layout-50ml.py:352 | FunctionDef | _augment_with_ml50
component/tokenisation_layout/tokenisation-layout.py:16 | FunctionDef | _get_pdf_reader
component/tokenisation_layout/tokenisation-layout.py:43 | FunctionDef | _ensure_nltk
component/tokenisation_layout/tokenisation-layout.py:65 | FunctionDef | detect_lang
component/tokenisation_layout/tokenisation-layout.py:94 | FunctionDef | split_ar_layout
component/tokenisation_layout/tokenisation-layout.py:107 | FunctionDef | _load_punkt_pickle
component/tokenisation_layout/tokenisation-layout.py:114 | FunctionDef | _split_sentence_fallback
component/tokenisation_layout/tokenisation-layout.py:127 | FunctionDef | split_punkt_layout
component/tokenisation_layout/tokenisation-layout.py:143 | FunctionDef | _split_on_newline_gap
component/tokenisation_layout/tokenisation-layout.py:162 | FunctionDef | _split_on_lang_switch
component/tokenisation_layout/tokenisation-layout.py:190 | FunctionDef | sentence_chunks_layout
component/tokenisation_layout/tokenisation-layout.py:210 | FunctionDef | _iter_line_spans
component/tokenisation_layout/tokenisation-layout.py:221 | FunctionDef | _collapse_ws
component/tokenisation_layout/tokenisation-layout.py:224 | FunctionDef | _mask_digits
component/tokenisation_layout/tokenisation-layout.py:251 | FunctionDef | _is_section_start_line
component/tokenisation_layout/tokenisation-layout.py:274 | FunctionDef | _merge_label_only
component/tokenisation_layout/tokenisation-layout.py:286 | FunctionDef | split_sections_layout
component/tokenisation_layout/tokenisation-layout.py:320 | FunctionDef | split_paragraphs_layout
component/tokenisation_layout/tokenisation-layout.py:335 | FunctionDef | chunk_layout_universal
component/tokenisation_layout/tokenisation-layout.py:405 | FunctionDef | _space_runs_ge
component/tokenisation_layout/tokenisation-layout.py:408 | FunctionDef | _has_big_gap
component/tokenisation_layout/tokenisation-layout.py:411 | FunctionDef | _num_tokens
component/tokenisation_layout/tokenisation-layout.py:414 | FunctionDef | _dec_tokens
component/tokenisation_layout/tokenisation-layout.py:417 | FunctionDef | _is_table_line
component/tokenisation_layout/tokenisation-layout.py:432 | FunctionDef | _cluster_centers
component/tokenisation_layout/tokenisation-layout.py:453 | FunctionDef | _upper_ratio
component/tokenisation_layout/tokenisation-layout.py:460 | FunctionDef | _sep_spans
component/tokenisation_layout/tokenisation-layout.py:479 | FunctionDef | _line_segments_by_gaps
component/tokenisation_layout/tokenisation-layout.py:500 | FunctionDef | _looks_like_title_line
component/tokenisation_layout/tokenisation-layout.py:512 | FunctionDef | _is_multicol_candidate_line
component/tokenisation_layout/tokenisation-layout.py:538 | FunctionDef | _looks_like_header_pair
component/tokenisation_layout/tokenisation-layout.py:548 | FunctionDef | _looks_like_addressish
component/tokenisation_layout/tokenisation-layout.py:558 | FunctionDef | _normalize_kv_generic
component/tokenisation_layout/tokenisation-layout.py:586 | FunctionDef | _strip_sep_lines
component/tokenisation_layout/tokenisation-layout.py:597 | FunctionDef | _assign_to_centers
component/tokenisation_layout/tokenisation-layout.py:609 | FunctionDef | _merge_close_columns
component/tokenisation_layout/tokenisation-layout.py:642 | FunctionDef | _is_grid_like
component/tokenisation_layout/tokenisation-layout.py:654 | FunctionDef | _is_micro_table_like
component/tokenisation_layout/tokenisation-layout.py:668 | FunctionDef | _transpose_or_group_multicol
component/tokenisation_layout/tokenisation-layout.py:845 | FunctionDef | _looks_like_paragraphish
component/tokenisation_layout/tokenisation-layout.py:855 | FunctionDef | _is_address_continuation_line
component/tokenisation_layout/tokenisation-layout.py:880 | FunctionDef | _collect_table_block
component/tokenisation_layout/tokenisation-layout.py:887 | FunctionDef | _looks_like_wrap_line
component/tokenisation_layout/tokenisation-layout.py:948 | FunctionDef | _make_span_item
component/tokenisation_layout/tokenisation-layout.py:961 | FunctionDef | layout_items
component/tokenisation_layout/tokenisation-layout.py:976 | FunctionDef | _starts_table
component/tokenisation_layout/tokenisation-layout.py:979 | FunctionDef | _starts_multicol
component/tokenisation_layout/tokenisation-layout.py:1084 | FunctionDef | _k
component/tokenisation_layout/tokenisation-layout.py:1100 | FunctionDef | build_noise_keys_for_doc
component/tokenisation_layout/tokenisation-layout.py:1143 | FunctionDef | chunk_is_noise
component/tokenisation_layout/tokenisation-layout.py:1172 | FunctionDef | _nonspace_len
component/tokenisation_layout/tokenisation-layout.py:1175 | FunctionDef | _line_col_from_offset
component/tokenisation_layout/tokenisation-layout.py:1186 | FunctionDef | _safe_str
component/tokenisation_layout/tokenisation-layout.py:1192 | FunctionDef | _unique_keep_order
component/tokenisation_layout/tokenisation-layout.py:1201 | FunctionDef | _pdf_extract_pages_text
component/tokenisation_layout/tokenisation-layout.py:1214 | FunctionDef | _pdf_page_count
component/tokenisation_layout/tokenisation-layout.py:1224 | FunctionDef | _safe_file_size
component/tokenisation_layout/tokenisation-layout.py:1233 | FunctionDef | _doc_size_from_paths
component/tokenisation_layout/tokenisation-layout.py:1468 | FunctionDef | _sort_key
component/tokenisation_layout/tokenisation-layout.py:1479 | FunctionDef | _select_doc
component/tokenisation_layout/tokenisation-layout.py:1506 | FunctionDef | print_one_doc
component/verification-totaux.py:13 | FunctionDef | _safe_list
component/verification-totaux.py:17 | FunctionDef | _safe_int
component/verification-totaux.py:24 | FunctionDef | _safe_float
component/verification-totaux.py:31 | FunctionDef | _to_decimal
component/verification-totaux.py:62 | FunctionDef | _money_str
component/verification-totaux.py:68 | FunctionDef | _is_close
component/verification-totaux.py:74 | FunctionDef | _norm_match_text
component/verification-totaux.py:82 | FunctionDef | _doc_aliases
component/verification-totaux.py:100 | FunctionDef | _build_chunk_lookup
component/verification-totaux.py:140 | FunctionDef | _value_position_in_text
component/verification-totaux.py:149 | FunctionDef | _locate_chunk
component/verification-totaux.py:214 | FunctionDef | _dominant_table_index
component/verification-totaux.py:226 | FunctionDef | _table_anchor_location
component/verification-totaux.py:254 | FunctionDef | _row_total_candidate
component/verification-totaux.py:262 | FunctionDef | _row_label
component/verification-totaux.py:270 | FunctionDef | _candidate_values
component/verification-totaux.py:285 | FunctionDef | _pick_nearest
component/verification-totaux.py:291 | FunctionDef | _pick_total_candidate
component/verification-totaux.py:309 | FunctionDef | _pick_subtotal_candidate
component/verification-totaux.py:316 | FunctionDef | _pick_tax_candidate
component/verification-totaux.py:329 | FunctionDef | _verify_row
component/verification-totaux.py:408 | FunctionDef | _doc_key
component/verification-totaux.py:418 | FunctionDef | _locate_total_value
component/verification-totaux.py:446 | FunctionDef | _verify_doc
component/verification-totaux.py:683 | FunctionDef | run
pipeline/cli.py:39 | FunctionDef | _step_choices
pipeline/cli.py:56 | FunctionDef | _env_bool
pipeline/cli.py:63 | FunctionDef | _env_int
pipeline/cli.py:73 | FunctionDef | _env_optional
pipeline/cli.py:81 | FunctionDef | _normalize_pipeline_name
pipeline/cli.py:85 | FunctionDef | _env_pipeline
pipeline/cli.py:93 | FunctionDef | _normalize_step_name
pipeline/cli.py:102 | FunctionDef | parse_cli
pipeline/cli.py:180 | FunctionDef | main
pipeline/cli.py:211 | FunctionDef | tee_print
pipeline/component_trace.py:9 | FunctionDef | _iso_now
pipeline/component_trace.py:13 | FunctionDef | sanitize_component_key
pipeline/component_trace.py:20 | FunctionDef | _safe_int
pipeline/component_trace.py:27 | FunctionDef | _json_safe
pipeline/component_trace.py:39 | FunctionDef | _fingerprint
pipeline/component_trace.py:59 | FunctionDef | capture_context_fingerprints
pipeline/component_trace.py:68 | FunctionDef | _ensure_component_traces
pipeline/component_trace.py:76 | FunctionDef | _pipeline_step_index
pipeline/component_trace.py:86 | FunctionDef | start_component_trace
pipeline/component_trace.py:108 | FunctionDef | finish_component_trace
pipeline/component_trace.py:133 | FunctionDef | report_component_trace
pipeline/component_trace.py:152 | FunctionDef | component_trace_public_rows
pipeline/components.py:39 | FunctionDef | _doc_has_meaningful_text
pipeline/components.py:66 | FunctionDef | _drop_empty_duplicate_docs
pipeline/components.py:91 | ClassDef | Component
pipeline/components.py:97 | FunctionDef | run
pipeline/components.py:119 | FunctionDef | _execute_script
pipeline/components.py:162 | FunctionDef | _report
pipeline/components.py:172 | ClassDef | PretraitementComponent
pipeline/components.py:173 | FunctionDef | run
pipeline/components.py:225 | ClassDef | OCRPreprocessComponent
pipeline/components.py:226 | FunctionDef | run
pipeline/components.py:279 | ClassDef | OutputTxtComponent
pipeline/components.py:280 | FunctionDef | run
pipeline/components.py:290 | FunctionDef | _doc_pages_count
pipeline/components.py:301 | ClassDef | TokenisationLayoutComponent
pipeline/components.py:302 | FunctionDef | run
pipeline/components.py:328 | ClassDef | GrammarComponent
pipeline/components.py:329 | FunctionDef | run
pipeline/components.py:370 | ClassDef | TableExtractionComponent
pipeline/components.py:371 | FunctionDef | run
pipeline/components.py:393 | ClassDef | TotalsVerificationComponent
pipeline/components.py:394 | FunctionDef | run
pipeline/components.py:413 | ClassDef | VisualMarksDetectionComponent
pipeline/components.py:414 | FunctionDef | run
pipeline/components.py:445 | ClassDef | InterDocLinkingComponent
pipeline/components.py:446 | FunctionDef | run
pipeline/components.py:477 | ClassDef | ElasticsearchComponent
pipeline/components.py:478 | FunctionDef | run
pipeline/components.py:518 | ClassDef | ClassificationComponent
pipeline/components.py:519 | FunctionDef | run
pipeline/components.py:565 | ClassDef | RuleExtractionComponent
pipeline/components.py:566 | FunctionDef | run
pipeline/components.py:615 | ClassDef | FusionResultComponent
pipeline/components.py:618 | FunctionDef | run
pipeline/components.py:661 | ClassDef | APIOutputComponent
pipeline/components.py:664 | FunctionDef | run
pipeline/elasticsearch.py:33 | FunctionDef | _iso_now
pipeline/elasticsearch.py:37 | FunctionDef | _unique_keep_order
pipeline/elasticsearch.py:48 | FunctionDef | _safe_int
pipeline/elasticsearch.py:55 | FunctionDef | _safe_positive_int
pipeline/elasticsearch.py:65 | FunctionDef | _is_local_es_url
pipeline/elasticsearch.py:70 | FunctionDef | _normalize_command
pipeline/elasticsearch.py:83 | FunctionDef | _format_command
pipeline/elasticsearch.py:87 | FunctionDef | _resolve_auto_start_commands
pipeline/elasticsearch.py:90 | FunctionDef | _append_many
pipeline/elasticsearch.py:120 | FunctionDef | _append_one
pipeline/elasticsearch.py:160 | FunctionDef | _run_auto_start_command
pipeline/elasticsearch.py:195 | FunctionDef | _wait_for_es_ping
pipeline/elasticsearch.py:204 | FunctionDef | _try_auto_start_elasticsearch
pipeline/elasticsearch.py:274 | FunctionDef | _same_es_target
pipeline/elasticsearch.py:280 | FunctionDef | _split_words
pipeline/elasticsearch.py:285 | FunctionDef | _json_serialize
pipeline/elasticsearch.py:292 | FunctionDef | _normalize_nlp_level
pipeline/elasticsearch.py:303 | FunctionDef | _normalize_nlp_index
pipeline/elasticsearch.py:311 | FunctionDef | _to_clean_str
pipeline/elasticsearch.py:316 | FunctionDef | _top_counter_items
pipeline/elasticsearch.py:320 | FunctionDef | _extract_entities
pipeline/elasticsearch.py:354 | FunctionDef | _entities_sample_flat
pipeline/elasticsearch.py:366 | FunctionDef | build_es_doc_id
pipeline/elasticsearch.py:379 | FunctionDef | _file_size_from_paths
pipeline/elasticsearch.py:405 | FunctionDef | _extract_doc_size
pipeline/elasticsearch.py:417 | FunctionDef | _page_text_from_page
pipeline/elasticsearch.py:439 | FunctionDef | flatten_tok_doc_for_index
pipeline/elasticsearch.py:525 | ClassDef | ElasticsearchStore
pipeline/elasticsearch.py:526 | FunctionDef | __init__
pipeline/elasticsearch.py:542 | FunctionDef | _request
pipeline/elasticsearch.py:588 | FunctionDef | ping
pipeline/elasticsearch.py:595 | FunctionDef | ensure_index
pipeline/elasticsearch.py:678 | FunctionDef | ensure_custom_index
pipeline/elasticsearch.py:690 | FunctionDef | delete_by_query
pipeline/elasticsearch.py:700 | FunctionDef | bulk_ndjson
pipeline/elasticsearch.py:712 | FunctionDef | upsert_document
pipeline/elasticsearch.py:719 | FunctionDef | update_fields
pipeline/elasticsearch.py:722 | FunctionDef | get_document
pipeline/elasticsearch.py:730 | FunctionDef | mget
pipeline/elasticsearch.py:744 | FunctionDef | search
pipeline/elasticsearch.py:755 | FunctionDef | find_document_id
pipeline/elasticsearch.py:774 | FunctionDef | index_tok_docs
pipeline/elasticsearch.py:791 | FunctionDef | _group_passages_by_page
pipeline/elasticsearch.py:804 | FunctionDef | to_classification_docs
pipeline/elasticsearch.py:847 | FunctionDef | to_extraction_docs
pipeline/elasticsearch.py:890 | FunctionDef | fetch_sources_for_ids
pipeline/elasticsearch.py:897 | FunctionDef | update_classification_results
pipeline/elasticsearch.py:924 | FunctionDef | update_extraction_results
pipeline/elasticsearch.py:970 | FunctionDef | _build_nlp_doc_lookup
pipeline/elasticsearch.py:990 | FunctionDef | _resolve_nlp_doc_id
pipeline/elasticsearch.py:1015 | FunctionDef | _ensure_nlp_tokens_index
pipeline/elasticsearch.py:1038 | FunctionDef | _build_nlp_token_doc_id
pipeline/elasticsearch.py:1049 | FunctionDef | _bulk_index_nlp_tokens
pipeline/elasticsearch.py:1079 | FunctionDef | sync_nlp_results
pipeline/elasticsearch.py:1254 | FunctionDef | maybe_build_store
pipeline/file_resolution.py:27 | FunctionDef | _sha256_path
pipeline/file_resolution.py:35 | FunctionDef | parse_git_lfs_pointer_bytes
pipeline/file_resolution.py:55 | FunctionDef | parse_git_lfs_pointer_path
pipeline/file_resolution.py:63 | FunctionDef | _iter_named_candidates
pipeline/file_resolution.py:79 | FunctionDef | resolve_git_lfs_pointer_path
pipeline/file_resolution.py:91 | FunctionDef | _resolve_pointer_spec
pipeline/file_resolution.py:122 | FunctionDef | resolve_runtime_input_path
pipeline/file_resolution.py:129 | FunctionDef | materialize_uploaded_content_from_lfs_pointer
pipeline/local_api.py:53 | FunctionDef | _active_pipeline_profile
pipeline/local_api.py:58 | FunctionDef | _active_pipeline_source
pipeline/local_api.py:66 | FunctionDef | _active_pipeline_steps
pipeline/local_api.py:71 | FunctionDef | _active_pipeline_metadata
pipeline/local_api.py:107 | FunctionDef | _iso_now
pipeline/local_api.py:111 | FunctionDef | _json_bytes
pipeline/local_api.py:115 | FunctionDef | _tail_recent_lines
pipeline/local_api.py:131 | FunctionDef | _metadata_from_snapshot
pipeline/local_api.py:144 | FunctionDef | _current_runtime_progress
pipeline/local_api.py:243 | FunctionDef | _discover_ipv4_addresses
pipeline/local_api.py:262 | FunctionDef | _candidate_urls
pipeline/local_api.py:277 | FunctionDef | _sanitize_filename
pipeline/local_api.py:284 | FunctionDef | _extract_uploaded_payload
pipeline/local_api.py:362 | FunctionDef | _sha256_bytes
pipeline/local_api.py:368 | FunctionDef | _request_origin
pipeline/local_api.py:374 | FunctionDef | _first_text_field
pipeline/local_api.py:384 | FunctionDef | _public_api_base_url
pipeline/local_api.py:390 | FunctionDef | _stored_manifest_path
pipeline/local_api.py:394 | FunctionDef | _stored_result_path
pipeline/local_api.py:398 | FunctionDef | _api_file_route
pipeline/local_api.py:402 | FunctionDef | _api_manifest_route
pipeline/local_api.py:406 | FunctionDef | _api_result_route
pipeline/local_api.py:410 | FunctionDef | _build_public_file_url
pipeline/local_api.py:417 | FunctionDef | _build_public_manifest_url
pipeline/local_api.py:424 | FunctionDef | _build_public_result_url
pipeline/local_api.py:431 | FunctionDef | _load_manifest
pipeline/local_api.py:441 | FunctionDef | _write_manifest
pipeline/local_api.py:448 | FunctionDef | _load_result
pipeline/local_api.py:458 | FunctionDef | _result_info
pipeline/local_api.py:471 | FunctionDef | save_uploaded_files
pipeline/local_api.py:582 | FunctionDef | _documents_index_payload
pipeline/local_api.py:599 | FunctionDef | _send_file_response
pipeline/local_api.py:617 | FunctionDef | build_cli_command
pipeline/local_api.py:626 | ClassDef | LauncherState
pipeline/local_api.py:627 | FunctionDef | __init__
pipeline/local_api.py:644 | FunctionDef | snapshot
pipeline/local_api.py:688 | FunctionDef | start_job
pipeline/local_api.py:757 | FunctionDef | _wait_for_process
pipeline/local_api.py:770 | ClassDef | DMSLauncherHandler
pipeline/local_api.py:774 | FunctionDef | launcher_state
pipeline/local_api.py:777 | FunctionDef | _send_cors_headers
pipeline/local_api.py:782 | FunctionDef | _send_json
pipeline/local_api.py:791 | FunctionDef | _send_html_file
pipeline/local_api.py:800 | FunctionDef | log_message
pipeline/local_api.py:803 | FunctionDef | do_GET
pipeline/local_api.py:869 | FunctionDef | do_OPTIONS
pipeline/local_api.py:874 | FunctionDef | do_POST
pipeline/local_api.py:984 | ClassDef | DMSLauncherServer
pipeline/local_api.py:987 | FunctionDef | __init__
pipeline/local_api.py:992 | FunctionDef | parse_args
pipeline/local_api.py:999 | FunctionDef | serve
pipeline/local_api.py:1019 | FunctionDef | _graceful_stop
pipeline/local_api.py:1040 | FunctionDef | main
pipeline/orchestrator.py:28 | ClassDef | BasePipelineOrchestrator
pipeline/orchestrator.py:34 | FunctionDef | __init__
pipeline/orchestrator.py:38 | FunctionDef | build_components
pipeline/orchestrator.py:41 | FunctionDef | list_steps
pipeline/orchestrator.py:44 | FunctionDef | _select_components
pipeline/orchestrator.py:62 | FunctionDef | run
pipeline/orchestrator.py:89 | FunctionDef | _pipeline_sort_key
pipeline/orchestrator.py:98 | FunctionDef | pipeline_orchestrator_classes
pipeline/orchestrator.py:114 | FunctionDef | pipeline_registry
pipeline/orchestrator.py:131 | FunctionDef | available_pipeline_codes
pipeline/orchestrator.py:135 | FunctionDef | available_pipeline_choices
pipeline/orchestrator.py:145 | FunctionDef | pipeline_definition
pipeline/orchestrator.py:154 | FunctionDef | normalize_pipeline_name
pipeline/orchestrator.py:185 | FunctionDef | create_pipeline_orchestrator
pipeline/orchestrator.py:191 | ClassDef | Pipeline0MLOrchestrator
pipeline/orchestrator.py:197 | FunctionDef | build_components
pipeline/orchestrator.py:215 | ClassDef | Pipeline50MLOrchestrator
pipeline/orchestrator.py:221 | FunctionDef | build_components
pipeline/orchestrator.py:239 | ClassDef | Pipeline100MLOrchestrator
pipeline/orchestrator.py:245 | FunctionDef | build_components
pipeline/runtime_state.py:14 | FunctionDef | _iso_now
pipeline/runtime_state.py:18 | FunctionDef | _state_path
pipeline/runtime_state.py:29 | FunctionDef | _steps
pipeline/runtime_state.py:36 | FunctionDef | _pipeline_profile
pipeline/runtime_state.py:45 | FunctionDef | _job_id
pipeline/runtime_state.py:54 | FunctionDef | read_runtime_state
pipeline/runtime_state.py:66 | FunctionDef | _write_runtime_state
pipeline/runtime_state.py:73 | FunctionDef | _step_index
pipeline/runtime_state.py:79 | FunctionDef | _running_progress_percent
pipeline/runtime_state.py:85 | FunctionDef | update_runtime_state
pipeline/runtime_state.py:105 | FunctionDef | publish_pipeline_started
pipeline/runtime_state.py:123 | FunctionDef | publish_pipeline_completed
pipeline/runtime_state.py:140 | FunctionDef | publish_pipeline_failed
pipeline/runtime_state.py:149 | FunctionDef | publish_component_started
pipeline/runtime_state.py:170 | FunctionDef | publish_component_completed
pipeline/runtime_state.py:191 | FunctionDef | publish_component_failed
pipeline/settings.py:20 | FunctionDef | load_dotenv
pipeline/settings.py:55 | FunctionDef | configure_logging
pipeline/settings.py:66 | FunctionDef | normalize_input
pipeline/settings.py:71 | FunctionDef | _split_item
pipeline/settings.py:84 | FunctionDef | safe_repr
pipeline/settings.py:89 | FunctionDef | count_sentences
pipeline/settings.py:98 | FunctionDef | change_dir
pipeline/settings.py:109 | FunctionDef | isolated_argv
pytesseract.py:14 | ClassDef | TesseractError
pytesseract.py:18 | ClassDef | TesseractNotFoundError
pytesseract.py:22 | ClassDef | Output
pytesseract.py:29 | FunctionDef | _split_config
pytesseract.py:39 | FunctionDef | _run_tesseract
pytesseract.py:60 | FunctionDef | _coerce_image_input
pytesseract.py:72 | FunctionDef | _cleanup_temp
pytesseract.py:81 | FunctionDef | get_tesseract_version
pytesseract.py:90 | FunctionDef | get_languages
pytesseract.py:98 | FunctionDef | image_to_osd
pytesseract.py:107 | FunctionDef | image_to_string
pytesseract.py:119 | FunctionDef | _parse_tsv_to_dict
pytesseract.py:133 | FunctionDef | image_to_data
```
