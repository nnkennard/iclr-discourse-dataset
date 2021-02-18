import collections
import json
import numpy as np
import os
import pickle
import torch

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi
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
  res =  query_idx[0] == doc_idx[0] and query_idx[1] == doc_idx[1]
  return res


def get_review_and_rebuttal_sentences(pair, pair_output_dir):
  sentences = []
  for comment_type in ["review", "rebuttal"]:
    comment_sentences = get_sentences_from_chunks(pair[comment_type + "_text"])
    with open(pair_output_dir + comment_type + ".txt", 'w') as f:
      f.write("\n".join([" ".join(sentence)
                         for sentence in comment_sentences]))
    sentences.append(comment_sentences)
  return sentences

def get_builder():
  traindev_datasets = [dataset
      for dataset in orl.DATASETS if 'traindev' in dataset]
  return collections.OrderedDict({
    dataset:collections.OrderedDict() for dataset in traindev_datasets 
    })

def texts_from_builder(builder):
  keys = []
  texts = []
  for dataset, dataset_dict in builder.items():
    for (pair, idx), text in dataset_dict.items():
      keys.append((dataset, pair, idx))
      texts.append(text)
  return keys, texts

def get_corpus_indices(query_key, corpus_keys):
  keep_indices = []
  for i, (c_dataset, pair_index, _) in enumerate(corpus_keys):
    if c_dataset == query_key[0] and pair_index == query_key[1]:
      keep_indices.append(i)

  assert len(keep_indices) == max(keep_indices) - min(keep_indices) + 1
  return min(keep_indices), max(keep_indices)


def main():

  data_dir = "../review_rebuttal_pair_dataset_debug/"
  output_dir = data_dir + "/ws/"
  dir_fix(output_dir)

  corpus_builder = get_builder()
  query_builder = get_builder()

  for dataset in orl.DATASETS:
    if 'traindev' not in dataset:
      continue
    input_file = data_dir + dataset + ".json"
    with open(input_file, 'r') as f:
      obj = json.load(f)

    for pair in tqdm(obj["review_rebuttal_pairs"]):
      pair_index = pair["index"]
      pair_output_dir = "/".join([output_dir, str(pair_index), ""])
      dir_fix(pair_output_dir)
      (review_sentences,
       rebuttal_sentences) = get_review_and_rebuttal_sentences(
           pair, pair_output_dir)
      for j, sentence in enumerate(review_sentences):
        corpus_builder[dataset][(pair_index, j)] = preprocess(sentence)
      for j, sentence in enumerate(rebuttal_sentences):
        query_builder[dataset][(pair_index, j)] = preprocess(sentence)

  corpus_keys, corpus = texts_from_builder(corpus_builder)
  query_keys, queries = texts_from_builder(query_builder)
  model = BM25Okapi(corpus)

  relevant_scores_map = collections.OrderedDict()
  for query_i, preprocessed_query in tqdm(enumerate(queries)):
    query_key = query_keys[query_i]
    min_idx, max_idx = get_corpus_indices(query_key, corpus_keys)
    scores = model.get_scores(preprocessed_query)
    relevant_scores = [score for score in range(min_idx, max_idx + 1)]
    relevant_scores_map[query_i] = np.array(relevant_scores)

  with open(output_dir + "scores.pickle", 'wb') as f:
    pickle.dump(relevant_scores_map, f)

  with open(output_dir + "keys.pickle", 'wb') as f:
    pickle.dump({
      "corpus_keys": corpus_keys,
      "query_keys": query_keys
      }, f)




if __name__ == "__main__":
  main()
