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
Result = collections.namedtuple("Result", "queries corpus scores".split())

STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')

def preprocess_sentence(sentence_tokens):
  return [STEMMER.stem(word).lower()
      for word in sentence_tokens
      if word.lower() not in STOPWORDS]


def get_top_k_indices(array, k):
  if k > len(array):
    top_k_list = list(enumerate(array))
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
      preprocess_sentence(sentence)))
  return documents


def get_key_prefix(obj):
  return "_".join([obj["split"], obj["subsplit"]])


def gather_datasets(data_dir):
  corpus_map = {}
  query_map = {}

  for dataset in orl.DATASETS:
    with open(data_dir + dataset + ".json", 'r') as f:
      obj = json.load(f)
    if dataset == orl.Split.UNSTRUCTURED:
      continue
    else:
      key_prefix = get_key_prefix(obj)
      queries = []
      corpus = []
      for pair in obj["review_rebuttal_pairs"][:100]:
        corpus += documents_from_chunks(pair["review_text"], key_prefix +
        "_review_" + str(pair["index"]))
        queries += documents_from_chunks(pair["rebuttal_text"], key_prefix +
        "_rebuttal_" + str(pair["index"]))
      query_map[dataset] = queries
      corpus_map[dataset] = corpus

  assert len(corpus_map) == len(query_map) == 4
  return corpus_map, query_map

PARTITION_K = 1000

def score_dataset(corpus, queries):
  model = BM25Okapi([doc.preprocessed_tokens for doc in corpus])
  scores = []
  for query in tqdm(queries):
    query_scores = model.get_scores(query.preprocessed_tokens).tolist()
    scores.append(get_top_k_indices(query_scores, PARTITION_K))
  return scores


def build_text_list(document_map):
  key_list = []
  texts = []
  for k in sorted(document_map.keys()):
    key_list.append(k)
    texts.append(document_map[k])
  return TextList(key_list, texts)


def score_datasets(corpus_map, query_map, data_dir):
  results = {}
  for dataset in orl.DATASETS:
    if dataset == orl.Split.UNSTRUCTURED:
      continue
    else:
      results[dataset] = score_dataset(corpus_map[dataset], query_map[dataset])
  return results


def write_datasets_to_file(corpus_map, query_map, data_dir):
  for dataset, corpus in corpus_map.items():
    sorted_queries = [q._asdict()
          for q in sorted(query_map[dataset], key=lambda x:x.key)]
    sorted_corpus = [d._asdict() for d in sorted(corpus, key=lambda x:x.key)]
    with open(data_dir + "/" + dataset +"_text.json", 'w') as f:
      json.dump({
        "corpus": sorted_corpus,
        "queries": sorted_queries
        }, f)

def main():
  data_dir = "../test_unlabeled/"
  corpus_map, query_map = gather_datasets(data_dir)
  # Dump datasets to file
  write_datasets_to_file(corpus_map, query_map, data_dir)
  bm25_scores = score_datasets(corpus_map, query_map, data_dir)
  for dataset, scores in bm25_scores.items():
    with open(data_dir + "/" + dataset +"_scores.json", 'w') as f:
      json.dump(scores, f)


if __name__ == "__main__":
  main()
