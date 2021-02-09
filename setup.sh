python3 -m venv iddve
source iddve/bin/activate
python -m pip install -r mini_requirements.txt

if [[ -z "$CORENLP_HOME" ]]; then
  echo "Please download Stanford CoreNLP and set \$CORENLP_HOME to point to the unzipped directory."
fi

python build_review_classification.py
python build_pair_datasets.py 
