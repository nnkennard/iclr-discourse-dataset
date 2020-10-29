# ICLR Discourse Dataset

You will need to have downloaded [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and set `$CORENLP_HOME` to point to the unzipped directory.

To set up the environment and build the database, run

```
bash setup.sh
```
This step takes about 30 minutes.

`example.py` can be used to dump examples to stdout.

```
source idd_ve/bin/activate
python example.py -n 5 # dump 5 examples
python example.py # dump all examples
```
