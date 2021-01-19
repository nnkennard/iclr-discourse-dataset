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
TextList = collections.namedtuple("TextList", "key_list texts".split())
Result = collections.namedtuple("Result", "queries corpus scores".split())

STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')

def preprocess_sentence(sentence_tokens):
  return [STEMMER.stem(word).lower()
      for word in sentence_tokens
      if word.lower() not in STOPWORDS]


def get_top_k_indices(array, k):
  if k > len(array):
    top_k_list(enumerate(array))
  else:
    neg_k = 0 - k
    indices = np.argpartition(array, neg_k)[neg_k:]
    top_k_list = [(i, array[i]) for i in indices]
  return list(
      reversed(sorted(
       top_k_list , key=lambda x:x[1])))



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

def score_dataset(queries, documents):
  model = BM25Okapi(documents.texts)
  scores = []
  for query in tqdm(queries.texts):
    query_scores = model.get_scores(query).tolist()
    scores.append(get_top_k_indices(query_scores, PARTITION_K))
  return scores


def build_text_list(document_map):
  key_list = []
  texts = []
  for k in sorted(document_map.keys()):
    key_list.append(k)
    texts.append(document_map[k])
  return TextList(key_list, texts)


def score_datasets(document_map, data_dir):
  results = {}
  for dataset in orl.DATASETS:
    if dataset == orl.Split.UNSTRUCTURED:
      continue
    else:
      documents = document_map[dataset]
      corpus = {}
      queries = {}
      for document in documents:
        key = document["key"]
        if 'rebuttal' in key:
          queries[key] = document["preprocessed_tokens"]
        else:
          assert 'review' in key
          corpus[key] = document["preprocessed_tokens"]
      query_obj = build_text_list(queries)
      corpus_obj = build_text_list(corpus)
      dataset_scores = score_dataset(query_obj, corpus_obj)
      results[dataset] = Result(query_obj._asdict(), corpus_obj._asdict(),
          dataset_scores)
  return results

def main():
  data_dir = "../test_unlabeled/"
  document_map = gather_datasets(data_dir)
  bm25_scores = score_datasets(document_map, data_dir)
  



if __name__ == "__main__":
  main()
