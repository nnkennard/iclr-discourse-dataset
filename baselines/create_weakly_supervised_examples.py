import json
import numpy as np
import sys


def main():
  filename = "bm25_scores/traindev_dev/bm25_scores.json"
  with open(filename, 'r') as f:
    bm25_scores = json.load(f)

  with open("bm25_inputs.json", 'r') as f:
    documents = json.load(f)["traindev_dev"]

  processed_text_map = {
      document["key"]: sorted(document["preprocessed_tokens"])
      for document in documents
      }

  raw_text_map = {
      document["key"]: document["tokens"]
      for document in documents
      }

  for i, score_array in enumerate(bm25_scores["scores"]):
    top_1000_indices = np.argpartition(np.array(score_array), -1000)[-1000:]
    if 'traindev_dev' not in bm25_scores["query_keys"][i]:
      continue
    else:
      query_key = bm25_scores["query_keys"][i]
      query = raw_text_map[query_key]
      preprocessed_query = processed_text_map[query_key]

    for j in top_1000_indices:

      if 'traindev_dev' not in bm25_scores["corpus_keys"][j]:
        continue
      else:
        print(score_array[j], preprocessed_query)
        print(raw_text_map[bm25_scores["corpus_keys"][j]])
        print(processed_text_map[bm25_scores["corpus_keys"][j]])
        print()


    if i == 10:
      break


if __name__ == "__main__":
  main()
