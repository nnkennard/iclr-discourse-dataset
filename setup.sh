#virtualenv --python `which python3` idd_ve
#source idd_ve/bin/activate
#python -m pip install -r requirements.txt

#rm -rf db
#mkdir db
#rm -rf temp
#mkdir temp
#python populate_database.py --debug
#python -c "import nltk; nltk.download('stopwords')"
cd baselines/rule_based
#python gather_matches.py
cd ../
python run_baselines.py

