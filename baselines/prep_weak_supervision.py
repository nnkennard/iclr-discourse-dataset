import collections
import json
import sys

import openreview_lib as orl

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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


def get_unstructured_documents(obj):
  key_prefix = get_key_prefix(obj)
  documents = []
  for i, chunks in enumerate(obj["review_rebuttal_text"]):
    documents += documents_from_chunks(chunks, key_prefix + "_comment_" + str(i))
  for i, chunks in enumerate(obj["abstracts"]):
    documents += documents_from_chunks(chunks, key_prefix + "_abstract_" +
        str(i))
  return documents


def get_structured_documents(obj):
  key_prefix = get_key_prefix(obj)
  documents = []
  for pair in obj["review_rebuttal_pairs"][:10]:
    documents += documents_from_chunks(pair["review_text"], key_prefix +
    "_review_" + str(pair["index"]))
    documents += documents_from_chunks(pair["rebuttal_text"], key_prefix +
    "_rebuttal_" + str(pair["index"]))
  return documents


def main():
  data_dir = "../unlabeled/"
  document_map = {}

  # Gather datasets
  for dataset in orl.DATASETS:
    print("*", dataset)
    with open(data_dir + dataset + ".json", 'r') as f:
      obj = json.load(f)
    if dataset == orl.Split.UNSTRUCTURED:
      continue
      documents = get_unstructured_documents(obj)
    else:
      documents = get_structured_documents(obj)
    document_map[dataset] = documents

  # Score with BM25



  with open("bm25_inputs_mini.json", 'w') as f:

    json.dump(document_map, f)



if __name__ == "__main__":
  main()
