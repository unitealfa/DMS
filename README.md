# DMS

## Installation

### 1. Installer Python
Télécharger et installer Python depuis :
https://www.python.org/downloads/

---

### 2. Installer Tesseract OCR
Télécharger l’installateur :
`tesseract-ocr-w64-setup-5.5.0.20241111.exe`

Dépôt GitHub :
https://github.com/smallpdf/tesseract

Après installation, ajouter les langues nécessaires dans le dossier des données Tesseract.

---

### 3. Installer les dépendances Python
```bash
pip install -r requirements.txt
```

---

### 4. Installer les modèles SpaCy
```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

---

### 5. Installer Elasticsearch
Télécharger la version 7.10.2 :
https://www.elastic.co/downloads/past-releases/elasticsearch-7-10-2

Extraire l’archive puis lancer Elasticsearch.

Un script `.bat` est fourni pour automatiser le démarrage.  
Modifier le chemin selon l’emplacement d’extraction, par exemple :
```
C:\Users\moura\Downloads\elasticsearch-7.10.2-windows-x86_64\elasticsearch-7.10.2
```

---

## Notes
- Vérifier que Python est ajouté au PATH.
- Vérifier que Tesseract est accessible depuis la ligne de commande.
- Installer uniquement les langues nécessaires pour éviter d’alourdir l’environnement.
