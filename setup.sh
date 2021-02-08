python3 -m venv iddve
source iddve/bin/activate
python -m pip install -r mini_requirements.txt

mkdir review_classification
python build_review_classification.py
