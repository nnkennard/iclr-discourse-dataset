import gensim
import math

import utils

import gensim.downloader as api
from gensim import similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary


class VectorSpaceModel(object):
  def __init__(self):
    pass

  def predict(self):
    pass


def cosine_denominator(vec):
  return math.sqrt(sum(math.pow(x, 2) for x in vec.values()))

def cosine(vec1, vec2):
  dvec1 = dict(vec1)
  dvec2 = dict(vec2)

  if not dvec1.keys() or not dvec2.keys():
    return 0.0

  num_acc = 0.0
  for k, v in dvec1.items():
    if k in dvec2:
      num_acc += v * dvec2[k]

  return num_acc / (cosine_denominator(dvec1) * cosine_denominator(dvec2))


class TfIdfModel(VectorSpaceModel):

  def __init__(self, datasets, discourse_unit=None):
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
    print("Attempting to predict")
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
          sim = cosine(b_vec, v_vec)
          if sim > max_cosine:
            max_cosine = sim
            max_index = i
        labels.append(max_index)
      predictions[example["rebuttal_sid"]] = labels

    return predictions

