import collections
import json
import numpy as np
import os
import pickle
import torch

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi
#from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import openreview_lib as orl

STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')


def preprocess(sentence_tokens):
  return [
      STEMMER.stem(word).lower() for word in sentence_tokens
      if word.lower() not in STOPWORDS
  ]


def dir_fix(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def get_sentences_from_chunks(chunks):
  sentences = sum(chunks, [])
  return [sent for sent in sentences if sent]


def is_relevant(query_idx, doc_idx):
  return query_idx[0] == doc_idx[0] and query_idx[1] == doc_idx[1]


def get_review_and_rebuttal_sentences(pair, pair_output_dir):
  sentences = []
  for comment_type in ["review", "rebuttal"]:
    comment_sentences = get_sentences_from_chunks(pair[comment_type + "_text"])
    with open(pair_output_dir + comment_type + ".txt", 'w') as f:
      f.write("\n".join([" ".join(sentence)
                         for sentence in comment_sentences]))
    sentences.append(comment_sentences)
  return sentences


def main():

  data_dir = "../review_rebuttal_pair_dataset/"
  output_dir = data_dir + "/ws/"
  dir_fix(output_dir)

  for dataset in orl.DATASETS:
    if 'traindev' not in dataset:
      continue
    input_file = data_dir + dataset + ".json"
    with open(input_file, 'r') as f:
      obj = json.load(f)

    corpus_builder = collections.OrderedDict()
    query_builder = collections.OrderedDict()
    for pair in tqdm(obj["review_rebuttal_pairs"]):
      pair_index = pair["index"]
      pair_output_dir = "/".join([output_dir, str(pair_index), ""])
      dir_fix(pair_output_dir)
      (review_sentences,
       rebuttal_sentences) = get_review_and_rebuttal_sentences(
           pair, pair_output_dir)
      for j, sentence in enumerate(review_sentences):
        corpus_builder[(dataset, pair_index, j)] = preprocess(sentence)
      for j, sentence in enumerate(rebuttal_sentences):
        query_builder[(dataset, pair_index, j)] = preprocess(sentence)

  model = BM25Okapi(corpus_builder.values())

  relevant_scores_map = collections.OrderedDict()
  for query_idx, preprocessed_query in tqdm(query_builder.items()):
    scores = model.get_scores(preprocessed_query)
    relevant_scores = []
    for doc_idx, score in zip(corpus_builder.keys(), scores.tolist()):
      if is_relevant(query_idx, doc_idx):
        relevant_scores.append(score)
    relevant_scores_map[query_idx] = np.array(relevant_scores)

  with open(output_dir + "scores.pickle", 'wb') as f:
    pickle.dump(relevant_scores_map, f)


if __name__ == "__main__":
  main()
