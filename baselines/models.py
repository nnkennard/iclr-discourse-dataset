import collections
import gensim
import torch
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
    predictions = collections.defaultdict(list)
    for example in self.test_dataset["examples"]:
      review_reps = [self.encode(text) for text in example["review_text"]]
      rebuttal_reps = [self.encode(text) for text in example["rebuttal_text"]]
      for rebuttal_rep in rebuttal_reps:
        top_3 = self._get_top_predictions(review_reps, rebuttal_rep)
        predictions[example["rebuttal_sid"]].append(top_3)
    return predictions

  def _get_top_predictions(self, review_reps, rebuttal_rep):
    sims = torch.FloatTensor([self.sim(review_rep, rebuttal_rep)
      for review_rep in review_reps])
    return torch.topk(sims, k=min(3,len(sims))).indices.data.tolist()


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

  def encode(self, tokens):
    return self.model[self.dct.doc2bow(tokens)]

  def sim(self, vec1, vec2):
    return utils.sparse_cosine(vec1, vec2)


  def old_predict(self):

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

