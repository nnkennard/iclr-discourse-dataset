#virtualenv --python `which python3` iddve
#source iddve/bin/activate
#python -m pip install -r requirements.txt

#export CORENLP_HOME="/home/nnayak/stanford-corenlp-full-2018-02-27/"

rm -rf db
mkdir db
rm -rf temp
mkdir temp
python 2_populate_database.py
#python -c "import nltk; nltk.download('stopwords')"
mkdir baselines/datasets
cd baselines/rule_based
python gather_matches.py
cd ../
python create_datasets.py
python run_baselines.py

