import collections
import gensim
import gensim.downloader as api
import numpy as np
import torch
import utils

from gensim.corpora import Dictionary
from gensim import similarities
from gensim.models import TfidfModel
from rank_bm25 import BM25Okapi
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
    return self._pick_top_from_sims(sims)

  def _pick_top_from_sims(self, sims):
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

  def encode(self, tokens):
    return self.model[self.dct.doc2bow(tokens)]

  def sim(self, vec1, vec2):
    return utils.sparse_cosine(vec1, vec2)


class SentenceBERTModel(Model):

  def __init__(self, datasets):
    self.test_dataset = datasets["test"]
    self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

  def encode(self, tokens):
    return self.model.encode(" ".join(tokens))

  def sim(self, vec1, vec2):
    return np.dot(vec1, vec2)


class BMModel(Model):

  def __init__(self, datasets):
    self.test_dataset = datasets["test"]

  def predict(self):
    predictions = collections.defaultdict(list)
    for example in self.test_dataset["examples"]:
      model = BM25Okapi(example["review_text"])
      labels = []
      for query in example["rebuttal_text"]:
        top_3 = self._pick_top_from_sims(
            torch.FloatTensor(model.get_scores(query)))
        predictions[example["rebuttal_sid"]].append(top_3)
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
      review_embedding_matrix = np.stack([self.model.encode(sentence)
                                          for sentence in to_encode])
      labels = []
      for rebuttal_sentence in example["rebuttal_text"]:
        embedding = self.model.encode(" ".join(rebuttal_sentence))
        labels.append(np.argmax(np.dot(review_embedding_matrix, embedding)))
      predictions[example["rebuttal_sid"]] = labels

    return predictions



