import collections
import json
import sys
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import openreview_lib as orl

Score = collections.namedtuple("Score", "query_key doc_key score")

def main():
  data_dir = "bm25_scores/"
  with open("bm25_inputs_mini.json", 'r') as f:
    document_map = json.load(f)

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
        scores.append(model.get_scores(query).tolist())
      with open(data_dir + "/" + dataset + "/bm25_scores_mini.json", 'w') as f:
        results = {"corpus_keys": corpus_keys,
                   "query_keys": query_keys,
                   "scores": scores}
        json.dump(results, f)


if __name__ == "__main__":
  main()

