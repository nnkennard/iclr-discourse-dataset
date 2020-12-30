import collections
import gensim
import gensim.downloader as api
import json
from nltk.corpus import stopwords
import numpy as np
import torch
import utils

from gensim.corpora import Dictionary
from gensim import similarities
from gensim.models import TfidfModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, RobertaForQuestionAnswering


EMPTY_CHUNK = ["<br>"]
STOP_WORDS = set(stopwords.words('english')).union({",", "."})

def get_text_from_example(example, key):
  return [t for t in example[key] if not t == EMPTY_CHUNK]

def get_review_text(example):
  return get_text_from_example(example, "review_text")

def get_rebuttal_text(example):
  return get_text_from_example(example, "rebuttal_text")

def token_indexizer(text, piece_type):
  token_map = {}
  offset = 0
  for i, piece in enumerate(text):
    if piece_type == "chunk":
      tokens = sum(piece, [])
    else:
      tokens = piece
    token_map[i] = [offset + j for j in range(len(piece))]
    offset += len(piece)
  return token_map

def reverse_token_indexizer(text, piece_type):
  reverse_map = []
  for i, piece in enumerate(text):
    if piece_type == "chunk":
      tokens = sum(piece, [])
    else:
      tokens = piece
    reverse_map += [i] * len(tokens)
  return reverse_map

def token_indexizer_2(text, piece_type):
  reverse_map = reverse_token_indexizer(text, piece_type)
  token_map = collections.defaultdict(list)
  for token_i, piece_i in enumerate(reverse_map):
    token_map[piece_i].append(token_i)
  return token_map


class Model(object):
  def __init__(self, hyperparam_dict={}):
    pass

  def predict(self):
    predictions = collections.defaultdict(list)
    for review_sid, example in self.test_dataset.items():
      review_text = get_review_text(example)
      review_reps = [self.encode(text) for text in review_text]
      rebuttal_reps = [self.encode(text) for text in
          get_rebuttal_text(example)]
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

  def __init__(self, datasets, hyperparam_dict={}):
    self.test_dataset = datasets["train"]
    train_dataset = sum([
      get_review_text(example)
      for example in (list(datasets["train"].values()) +
      list(datasets["dev"].values()))],
      [])
    self.dct = Dictionary(train_dataset)
    corpus = [self.dct.doc2bow(line) for line in train_dataset]
    self.model = TfidfModel(corpus)

  def encode(self, tokens):
    return self.model[self.dct.doc2bow(tokens)]

  @staticmethod
  def sim(vec1, vec2):
    return utils.sparse_cosine(vec1, vec2)


class SentenceBERTModel(Model):

  def __init__(self, datasets, hyperparam_dict={}):
    self.test_dataset = datasets["train"]
    self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

  def encode(self, tokens):
    return self.model.encode(" ".join(tokens))

  @staticmethod
  def sim( vec1, vec2):
    return np.dot(vec1, vec2)/(
        np.linalg.norm(vec1)*np.linalg.norm(vec2))


class BMModel(Model):

  def __init__(self, datasets, hyperparam_dict={}):
    self.test_dataset = datasets["train"]

  def predict(self):
    predictions = collections.defaultdict(list)
    for review_sid, example in self.test_dataset.items():
      model = BM25Okapi(get_review_text(example))
      labels = []
      for query in get_rebuttal_text(example):
        top_3 = self._pick_top_from_sims(
            torch.FloatTensor(model.get_scores(query)))
        predictions[review_sid].append(top_3)
    return predictions
  

class RobertaModel(Model):

  def __init__(self, datasets, hyperparam_dict={}):
    self.test_dataset = datasets["train"]
    
    self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    self.model = RobertaForQuestionAnswering.from_pretrained('roberta-base')


  def predict(self):
    predictions = collections.defaultdict(list)
    for review_sid, example in self.test_dataset.items():
      review_text = get_review_text(example)
      text = " ".join(sum(review_text, []))[:512]
      for i, rebuttal_chunk in enumerate(get_rebuttal_text(example)):
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
          review_text, start_scores, offset_mapping)])
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

def make_content_set(tokens):
  k = set([token.lower() for token in list(tokens)
          if token.lower() not in STOP_WORDS])
  return k


def jaccard_similarity(set_1, set_2):
  return len(set_1.intersection(set_2))/len(set_1.union(set_2))

class RuleBasedModel(Model):
  def __init__(self, datasets, hyperparameter_dict={}):
    self.test_dataset = datasets["train"]
    hyperparameter_dict = {"match_file": "rule_based/matches_traindev_2162.json",
                           "piece_type": "sentence"}
    self.matches = self._get_matches_from_file(
        hyperparameter_dict["match_file"])

  def predict(self):
    predictions = collections.defaultdict(list)
    for review_sid, example in self.test_dataset.items():
      review_text = get_review_text(example)
      rebuttal_text = get_rebuttal_text(example)
      results = self._score_review_pieces(
          review_sid, rebuttal_pieces=rebuttal_text, review_pieces=review_text)

  def _jaccard_index(self, review_pieces, rebuttal_pieces):

    review_sets = [make_content_set(piece) for piece in review_pieces]
    rebuttal_sets = [make_content_set(piece) for piece in rebuttal_pieces]

    jaccard_map = collections.defaultdict(dict)
    for i, rebuttal_set in enumerate(rebuttal_sets):
      for j, review_set in enumerate(review_sets):
        jaccard_map[i][j] = jaccard_similarity(rebuttal_set, review_set)

    return jaccard_map


  def _score_review_pieces(self, review_sid, rebuttal_pieces, review_pieces):
    rebuttal_token_map = token_indexizer(rebuttal_pieces, "sentence")

    jaccard_map = self._jaccard_index(review_pieces=review_pieces,
        rebuttal_pieces=rebuttal_pieces)

    for i, rebuttal_piece in enumerate(rebuttal_pieces):
      relevant_matches = [match
          for match in self.matches[review_sid]
          if match["rebuttal_start"] in rebuttal_token_map[i]]
      if relevant_matches:
        pass
      else: # No exact matches 2 or more tokens long; backoff to Jaccard index
        similarities = jaccard_map[i]


  def _get_matches_from_file(self, filename):
    with open(filename, 'r') as f:
      return json.load(f)



