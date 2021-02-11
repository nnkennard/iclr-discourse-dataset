import collections
import json
import numpy as np
import random
import sys

import openreview_lib as orl

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import openreview_lib as orl

random.seed(34)


Document = collections.namedtuple("Document",
                                  "key tokens preprocessed_tokens".split())
Result = collections.namedtuple("Result", "queries corpus scores".split())

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
      for pair in obj["review_rebuttal_pairs"]:
        corpus += documents_from_chunks(pair["review_text"], key_prefix +
        "_review_" + str(pair["index"]))
        queries += documents_from_chunks(pair["rebuttal_text"], key_prefix +
        "_rebuttal_" + str(pair["index"]))
        if len(corpus) > 10000:
          break
      query_map[dataset] = list(sorted(queries, key=lambda x:x.key))
      corpus_map[dataset] = list(sorted(corpus, key=lambda x:x.key))

  assert len(corpus_map) == len(query_map) == 4
  return corpus_map, query_map

NUM_SAMPLES = 30

def score_dataset(corpus, queries):
  model = BM25Okapi([doc.preprocessed_tokens for doc in corpus])
  scores = []
  for query in tqdm(queries):
    search_prefix = "_".join(query.key.replace("rebuttal",
        "review").split("_")[:4])
    relevant_doc_indices = [
          i for i, doc in enumerate(corpus) if doc.key.startswith(search_prefix)
          ]

    query_scores = model.get_scores(query.preprocessed_tokens).tolist()
    scores.append(sample_from_scores(query_scores, relevant_doc_indices,
      NUM_SAMPLES))
  return scores

def sample_from_scores(query_scores, relevant_doc_indices, num_samples):
  if len(relevant_doc_indices) > NUM_SAMPLES:
    sample_indices = []
  else:
    sample_indices = random.sample(list(range(len(query_scores))), NUM_SAMPLES -
        len(relevant_doc_indices))
  all_indices = set(sample_indices + relevant_doc_indices)
  return sorted([(i, query_scores[i]) for i in all_indices], key=lambda x:x[1])


def score_datasets_and_write(corpus_map, query_map, data_dir):
  results = {}
  for dataset in orl.DATASETS:
    if dataset == orl.Split.UNSTRUCTURED:
      continue
    else:
      results[dataset] = score_dataset(corpus_map[dataset], query_map[dataset])
  return results


def write_datasets_to_file(corpus_map, query_map, data_dir):
  for dataset, corpus in corpus_map.items():
    queries = query_map[dataset]
    with open(data_dir + "/" + dataset +"_text.json", 'w') as f:
      json.dump({
        "corpus": corpus,
        "queries": queries
        }, f)


def create_example_lines(split, corpus, queries, query_i, doc_1_i, doc_2_i):
  return "".join([
      "\t".join([split,
      str(query_i),
      str(doc_1_i),
      str(doc_2_i), "0"]), "\n",
      "\t".join([split,
      str(query_i),
      str(doc_2_i),
      str(doc_1_i), "1"]), "\n"])


def create_weak_supervision_tsv(results, data_dir, corpus_map, query_map):
  example_map = {}
  with open(data_dir + "/examples.tsv", 'w') as f:
    for dataset, dataset_results in results.items():
      if dataset == "traindev_train":
        split = 'train'
      elif dataset == "traindev_dev":
        split = 'dev'
      else:
        continue
      example_tuples = []
      corpus = corpus_map[dataset]
      queries = query_map[dataset]
      for query_i, scores in enumerate(tqdm(dataset_results)):
        for j, (doc_1_i, score_1) in enumerate(scores):
          for doc_2_i, score_2 in scores[j+1:]:
            f.write(create_example_lines(split, corpus, queries, query_i, doc_1_i, doc_2_i))


def create_weak_supervision_examples_and_write(results, data_dir):
  example_map = {}
  for dataset, dataset_results in results.items():
    example_tuples = []
    for query_i, scores in enumerate(tqdm(dataset_results)):
      for j, (doc_1_i, score_1) in enumerate(scores):
        for doc_2_i, score_2 in scores[j+1:]:
          if doc_2_i == doc_1_i:
            dsds
          example_tuples.append(Example(query_i, doc_1_i, doc_2_i, 0))
          example_tuples.append(Example(query_i, doc_2_i, doc_1_i, 1))
    with open(data_dir + "/" + dataset +"_examples.json", 'w') as f:
      json.dump(example_tuples, f)
    example_map[dataset] = example_tuples
  return example_map


def main():
  data_dir = "../review_rebuttal_pair_dataset_debug/"
  corpus_map, query_map = gather_datasets(data_dir)
  write_datasets_to_file(corpus_map, query_map, data_dir)
  results = score_datasets_and_write(corpus_map, query_map, data_dir)
  examples = create_weak_supervision_tsv(
      results, data_dir, corpus_map, query_map)
  
if __name__ == "__main__":
  main()
