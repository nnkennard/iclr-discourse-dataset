python3 -m venv iddve
source iddve/bin/activate
python -m pip install -r mini_requirements.txt

python build_review_classification.py
python build_pair_datasets.py 
