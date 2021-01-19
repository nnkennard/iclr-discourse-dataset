import collections
import json
import numpy as np
import sys

import openreview_lib as orl

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import openreview_lib as orl


Document = collections.namedtuple("Document",
"key tokens preprocessed_tokens".split())

STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')

def preprocess_sentence(sentence_tokens):
  return [STEMMER.stem(word).lower()
      for word in sentence_tokens
      if word.lower() not in STOPWORDS]


def documents_from_chunks(chunks, key_prefix):
  sentences = []
  for chunk in chunks:
    for sentence in chunk:
      if not sentence:
        continue
      sentences.append(sentence)
  documents = []
  for i, sentence in enumerate(sentences):
    key = "_".join([key_prefix, str(i)])
    documents.append(Document(key, sentence,
      preprocess_sentence(sentence))._asdict())
  return documents


def get_key_prefix(obj):
  return "_".join([obj["split"], obj["subsplit"]])


def get_structured_documents(obj):
  key_prefix = get_key_prefix(obj)
  documents = []
  for pair in obj["review_rebuttal_pairs"][:100]:
    documents += documents_from_chunks(pair["review_text"], key_prefix +
    "_review_" + str(pair["index"]))
    documents += documents_from_chunks(pair["rebuttal_text"], key_prefix +
    "_rebuttal_" + str(pair["index"]))
  return documents


def gather_datasets(data_dir):
  document_map = {}

  for dataset in orl.DATASETS:
    with open(data_dir + dataset + ".json", 'r') as f:
      obj = json.load(f)
    if dataset == orl.Split.UNSTRUCTURED:
      continue
    else:
      documents = get_structured_documents(obj)
    document_map[dataset] = documents

  return document_map

PARTITION_K = 1000


def score_documents(document_map, data_dir):
  for dataset in orl.DATASETS:
    print(dataset)
    if dataset == orl.Split.UNSTRUCTURED:
      continue
    else:
      documents = document_map[dataset]
      corpus_keys = []
      corpus = []
      query_keys = []
      queries = []
      scores = []
      for document in documents:
        key = document["key"]
        if 'rebuttal' in key:
          query_keys.append(key)
          queries.append(document["preprocessed_tokens"])
        else:
          assert 'review' in key
          corpus_keys.append(key)
          corpus.append(document["preprocessed_tokens"])

      query_keys = sorted(query_keys)
      model = BM25Okapi(corpus)
      for query in tqdm(query_keys):
        query_scores = model.get_scores(query).tolist()
        scores.append(get_top_k_indices(query_scores, PARTITION_K))
      with open(data_dir + "/" + dataset + "_bm25_scores_mini.json", 'w') as f:
        results = {"corpus_keys": corpus_keys,
                   "query_keys": query_keys,
                   "scores": scores}
        return results

def get_top_k_indices(array, k):
  if k > len(array):
    return list(enumerate(array))
  neg_k = 0 - k
  indices = np.argpartition(array, neg_k)[neg_k:]
  return [(i, array[i]) for i in indices]

def main():
  data_dir = "../test_unlabeled/"
  document_map = gather_datasets(data_dir)
  print(document_map["truetest"][0])
  bm25_scores = score_documents(document_map, data_dir)




if __name__ == "__main__":
  main()
