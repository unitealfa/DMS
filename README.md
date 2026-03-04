DMS
Installation
🔹 Installation sous Ubuntu 22.04

Vérifier Python :

python3 --version

Si Python n’est pas installé :

sudo apt update
sudo apt install python3 python3-venv python3-pip

Installer les dépendances nécessaires pour l’OCR et les PDF :

sudo apt install tesseract-ocr poppler-utils

Installer les langues Tesseract nécessaires (optionnel mais recommandé) :

sudo apt install tesseract-ocr-fra tesseract-ocr-eng tesseract-ocr-ara

Vérifier Tesseract :

tesseract --version
2) Placer le projet et entrer dans le dossier

Se placer dans le dossier du projet (où se trouve requirements.txt) :

cd chemin/vers/le/dossier/DMS

Exemple :

cd ~/Bureau/DMS

sudo apt update
sudo apt install python3-venv

3) Créer un environnement virtuel (fortement recommandé)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
4) Installer les dépendances Python
python -m pip install -r requirements.txt
5) Installer les modèles SpaCy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
6) Vérification
python -c "import sys; print('Python utilisé :', sys.executable)"
python -m pip list
7) Lancer le projet
source .venv/bin/activate
python orchestre.py "documents/mon_fichier.pdf"
🔹 Installation sous Windows
1. Installer Python

Télécharger et installer Python depuis :
https://www.python.org/downloads/

Important : cocher “Add Python to PATH” pendant l’installation.

Vérifier :

python --version
2. Installer Tesseract OCR

Télécharger l’installateur :

tesseract-ocr-w64-setup-5.5.0.20241111.exe

Dépôt GitHub :
https://github.com/smallpdf/tesseract

Après installation :

Ajouter les langues nécessaires dans le dossier tessdata

Vérifier :

tesseract --version
3. Installer les dépendances Python

Depuis le dossier du projet :

pip install -r requirements.txt
4. Installer les modèles SpaCy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
5. Installer Elasticsearch

Télécharger la version 7.10.2 :
https://www.elastic.co/downloads/past-releases/elasticsearch-7-10-2

Extraire l’archive.

Lancer Elasticsearch depuis le dossier extrait.

Un script .bat peut être utilisé pour automatiser le démarrage.

Exemple de chemin Windows :

C:\Users\moura\Downloads\elasticsearch-7.10.2-windows-x86_64\elasticsearch-7.10.2
Notes importantes

Vérifier que Python est bien ajouté au PATH.

Vérifier que Tesseract est accessible en ligne de commande.

Installer uniquement les langues nécessaires pour éviter d’alourdir l’environnement.

Sous Linux, utiliser un environnement virtuel est fortement recommandé.

Tableau comparatif offline vs online (scores estimés)
Langue	POS offline	POS online	Δ POS	Lemma offline	Lemma online	Δ Lemma	NER offline	NER online	Δ NER
EN	76%	76%	0	79%	79%	0	38%	71%	+33
FR	66%	67%	+1	72%	73%	+1	52%	93%	+41
AR	53%	54%	+1	44%	45%	+1	74%	70%