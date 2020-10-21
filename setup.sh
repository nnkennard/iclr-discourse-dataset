virtualenv --python `which python3` idd_ve
source idd_ve/bin/activate
python -m pip install -r requirements.txt

mkdir db
python populate_database.py
