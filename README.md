# ICLR Discourse Dataset

You will need to have downloaded [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and set `$CORENLP_HOME` to point to the unzipped directory.

To set up the environment and build the dataset, run

```
bash setup.sh
```

This step takes about 30 minutes, and should produce dataset files in the directory `review_classification_dataset/` and `review_rebuttal_pair_dataset/`.

Output files are in JSON format. Text is represented as a list of list of
lists:

The top level lists represent chunks (‘paragraphs’ separated by new
lines). Each chunk is a list of sentences, and each sentence is a list of
tokens. Sentence splitting and tokenizing is carried out by the CoreNLP pipeline


