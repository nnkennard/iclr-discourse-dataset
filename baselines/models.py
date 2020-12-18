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
from transformers import AutoTokenizer, RobertaForQuestionAnswering


class Model(object):
  def __init__(self):
    pass

  def predict(self):
    predictions = collections.defaultdict(list)
    for review_sid, example in self.test_dataset.items():
      review_reps = [self.encode(text) for text in example["review_text"]]
      rebuttal_reps = [self.encode(text) for text in example["rebuttal_text"]]
      for rebuttal_rep in rebuttal_reps:
        top_3 = self._get_top_predictions(review_reps, rebuttal_rep)
        predictions[review_sid].append(top_3)
    return predictions

  def _get_top_predictions(self, review_reps, rebuttal_rep):
    sims = torch.FloatTensor([self.sim(review_rep, rebuttal_rep)
      for review_rep in review_reps])
    return self._pick_top_from_sims(sims)

  def _pick_top_from_sims(self, sims):
    return torch.topk(sims, k=min(3,len(sims))).indices.data.tolist()

class TfIdfModel(Model):

  def __init__(self, datasets):
    self.test_dataset = datasets["train"]
    train_dataset = sum([
      example["review_text"]
      for example in (list(datasets["train"].values()) +
      list(datasets["dev"].values()))],
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
    self.test_dataset = datasets["train"]
    self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

  def encode(self, tokens):
    return self.model.encode(" ".join(tokens))

  def sim(self, vec1, vec2):
    return np.dot(vec1, vec2)


class BMModel(Model):

  def __init__(self, datasets):
    self.test_dataset = datasets["train"]

  def predict(self):
    predictions = collections.defaultdict(list)
    for review_sid, example in self.test_dataset.items():
      model = BM25Okapi(example["review_text"])
      labels = []
      for query in example["rebuttal_text"]:
        top_3 = self._pick_top_from_sims(
            torch.FloatTensor(model.get_scores(query)))
        predictions[review_sid].append(top_3)
    return predictions
  

class RobertaModel(Model):

  def __init__(self, datasets):
    self.test_dataset = datasets["train"]
    
    self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    self.model = RobertaForQuestionAnswering.from_pretrained('roberta-base')


  def predict(self):
    predictions = collections.defaultdict(list)
    for review_sid, example in self.test_dataset.items():
      text = " ".join(sum(example["review_text"], []))[:512]
      for i, rebuttal_chunk in enumerate(example["rebuttal_text"]):
        question = " ".join(rebuttal_chunk)
        inputs = self.tokenizer(question, text, return_tensors='pt', 
                return_offsets_mapping=True)
        offset_mapping = inputs["offset_mapping"]
        del inputs["offset_mapping"]
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])

        outputs = self.model(**inputs, start_positions=start_positions,
            end_positions=end_positions)

        loss = outputs.loss
        start_scores = outputs.start_logits
        predictions[review_sid].append([self._get_predicted_chunk(
          example["review_text"], start_scores, offset_mapping)])
    return predictions


  def _get_predicted_chunk(self, chunks, start_scores, offset_mapping):
    top_logit = torch.argmax(start_scores)
    start_char, end_char = offset_mapping[0][top_logit]
    char_counter = 0
    for i, chunk in enumerate(chunks):
      text = " ".join(chunk)
      if start_char < char_counter + len(text):
        return i
      char_counter += len(text)
    assert False
        
    
    return predictions



