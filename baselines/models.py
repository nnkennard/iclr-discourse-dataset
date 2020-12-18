import gensim
from rank_bm25 import BM25Okapi

import utils

import gensim.downloader as api
from gensim import similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np

from sentence_transformers import SentenceTransformer

from transformers import RobertaTokenizer, RobertaModel

class Model(object):
  def __init__(self):
    pass

  def predict(self):
    pass

class TfIdfModel(Model):

  def __init__(self, datasets):
    self.test_dataset = datasets["test"]
    train_dataset = sum([
      example["review_text"]
      for example in (datasets["train"]["examples"] +
      datasets["dev"]["examples"])],
      [])
    self.dct = Dictionary(train_dataset)
    corpus = [self.dct.doc2bow(line) for line in train_dataset]
    self.model = TfidfModel(corpus)
    vector = self.model[corpus[0]]

  def predict(self):
    predictions = {}
    for example in self.test_dataset["examples"]:
      review_vectors = [self.model[self.dct.doc2bow(line)]
                        for line in example["review_text"]]
      rebuttal_vectors = [self.model[self.dct.doc2bow(line)]
                        for line in example["rebuttal_text"]]

      labels = []
      for b_vec in rebuttal_vectors:
        max_cosine = 0.0
        max_index = None
        for i, v_vec in enumerate(review_vectors):
          sim = utils.sparse_cosine(b_vec, v_vec)
          if sim > max_cosine:
            max_cosine = sim
            max_index = i
        labels.append(max_index)
      predictions[example["rebuttal_sid"]] = labels

    return predictions


class SentenceBERTModel(Model):

  def __init__(self, datasets):
    self.test_dataset = datasets["test"]
    self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

  def predict(self):
    predictions = {}
    for example in self.test_dataset["examples"]:
      to_encode = [" ".join(tokens) for tokens in example["review_text"]]
      review_embedding_matrix = np.stack([self.model.encode(sentence) for sentence in
        to_encode])
      labels = []
      for rebuttal_sentence in example["rebuttal_text"]:
        embedding = self.model.encode(" ".join(rebuttal_sentence))
        labels.append(np.argmax(np.dot(review_embedding_matrix, embedding)))
      predictions[example["rebuttal_sid"]] = labels

    return predictions

  
class RobertaModel(Model):

  def __init__(self, datasets):
    self.test_dataset = datasets["test"]
    
    self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    self.model = RobertaModel.from_pretrained('roberta-large')


  def predict(self):
    predictions = {}
    for example in self.test_dataset["examples"]:
      to_encode = [" ".join(tokens) for tokens in example["review_text"]]
      review_embedding_matrix = np.stack([self.model.encode(sentence) for sentence in
        to_encode])
      labels = []
      for rebuttal_sentence in example["rebuttal_text"]:
        embedding = self.model.encode(" ".join(rebuttal_sentence))
        labels.append(np.argmax(np.dot(review_embedding_matrix, embedding)))
      predictions[example["rebuttal_sid"]] = labels

    return predictions


class BMModel(Model):

  def __init__(self, datasets):
    self.test_dataset = datasets["test"]

  def predict(self):
    predictions = {}
    for example in self.test_dataset["examples"]:
      model = BM25Okapi(example["review_text"])
      labels = []
      for query in example["rebuttal_text"]:
        scores = model.get_scores(query)
        labels.append(scores.argmax())
      predictions[example["rebuttal_sid"]] = labels
    return predictions

