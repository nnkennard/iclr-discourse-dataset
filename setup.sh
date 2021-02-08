#virtualenv --python `which python3` iddve
#source iddve/bin/activate
#python -m pip install -r requirements.txt

export CORENLP_HOME="/home/nnayak/stanford-corenlp-full-2018-02-27/"
# ^ replace with path to your local CoreNLP directory

python3 -m venv iddve
source iddve/bin/activate
python -m pip install -r requirements.txt

mkdir review_classification
python build_review_classification.py
