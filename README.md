# ICLR Discourse Dataset

## Setup

0. If you are in IESL and using blake.cs.umass.edu, please run `module load python3/3.9.1-2102` first, without which it is possible that Stanza won't work.

1. Set up a Python virtual environment and download Stanza models (So now you don't have to do the CoreNLP stuff)
```
python3 -m venv iddve
source iddve/bin/activate
python -m pip install -r requirements/mini_requirements.txt

python -c "import stanza; stanza.download('en')"
```

2. Run code to create datasets
```
python build_pair_datasets.py
```

To build a smaller version of the dataset for viewing and testing, add `--debug`:

```
python build_pair_datasets.py --debug
```

This will create smaller datasets, and add them in a folder whose name ends in `_debug`.

3. Verify the built datasets

Run
```
python check.py
```
or 
```
python check.py --debug
```
depending on whether you have built the whole dataset or just the debug subset. If you don't get 'OK' for all the files... uhh, for now, ask Neha what to do about it

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
└─── review_rebuttal_pair_dataset_debug/ # if you ran with --debug as well
│   │   unstructured.json # These files will be much smaller
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
