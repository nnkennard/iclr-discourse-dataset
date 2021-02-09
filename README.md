# ICLR Discourse Dataset

## Setup

You will need to have downloaded [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and set `$CORENLP_HOME` to point to the unzipped directory.

To set up the environment and build the dataset, run

```
bash setup.sh
```

This step takes about 30 minutes, and should produce dataset files in the directory `review_classification_dataset/` and `review_rebuttal_pair_dataset/`.

## Data format


Output files are in JSON format. You should see this file structure:

```
iclr-discourse-dataset
│
└─── review_rebuttal_pair_dataset/
│   │   unstructured.json
│   │   traindev_train.json
│   │   traindev_dev.json
│   │   traindev_test.json
│   │   truetest.json
│   ... other ...
│   ... stuff ...

```
* Unstructured: unstructured text from reviews, rebuttals and abstracts, in *ICLR 2018*, for use in domain pre-training à la [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964)
* Truetest: (20% of all) review-rebuttal pairs from *ICLR 2020*, to be used as an unseen test set
* Traindev: review-rebuttal pairs from *ICLR 2019* in a traditional train/dev/test split. (3:1:1)

Each file has the following fields:
* `conference`: Which ICLR conference the examples are drawn from
* `split`: which split this data is from, out of unstructured/traindev/truetest
* `subsplit`: train, dev, or test
* `review_rebuttal_pairs`: a list of review-rebuttal pairs

Each review-rebuttal pair has the following fields:
* `index`: index within dataset 
* `review_sid`: 'super id' (id of first comment) in review 
* `rebuttal_sid`: 'super id' (id of first comment) in rebuttal 
* `review_text`: review text in chunks
* `rebuttal_text`: rebuttal text in chunks
* `title`: paper title
* `review_author`: id of the reviewer, e.g. "AnonReviewer1"
* `forum`: unique 'forum' id from OpenReview API -- identifies the paper
* `labels`: categorical labels where available, e.g. review rating, reviewer confidence

Text is represented as a list of list of lists:

The top level lists represent chunks (‘paragraphs’ separated by newlines). Each chunk is a list of sentences, and each sentence is a list of tokens. Sentence splitting and tokenizing is carried out by the CoreNLP pipeline.
