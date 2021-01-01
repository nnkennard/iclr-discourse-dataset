import argparse
import collections
import json
import torch

from transformers import RobertaTokenizer, RobertaForQuestionAnswering

import openreview_db as ordb
import models
import utils

from tqdm import tqdm


parser = argparse.ArgumentParser(
        description='Create datasets for training baseline models')
#parser.add_argument('-b', '--dbfile', default="../db/or.db",
#        type=str, help="Database in sqlite3 format")
parser.add_argument('-d', '--data_dir', default="datasets/",
        type=str, help="Dataset directory")
parser.add_argument('-r', '--result_dir', default="results/",
        type=str, help="Results directory")
parser.add_argument('-n', '--numexamples', default=-1,
        type=int, help="Number of examples per dataset; -1 to include all")


MODEL_MAP = {
  #"tfidf": models.TfIdfModel,
  #"sbert": models.SentenceBERTModel,
  #"roberta": models.RobertaModel,
  #"bm25": models.BMModel,
  "rule": models.RuleBasedModel,
    }

def load_dataset_splits(data_dir, discourse_unit):
  datasets = {}
  for split in ["train", "dev", "test"]:
    with open(utils.get_dataset_filename(data_dir, "traindev", split
      ), 'r') as f:
      obj = json.load(f)
      datasets[split] = {example["review_sid"]: example
          for example in obj["examples"]}
  return datasets 

EMPTY_CHUNK = ["<br>"]

def print_with_votes(example, vote_map, voter_map):
  for i, rebuttal_chunk in enumerate(example["rebuttal_text"]):
    if rebuttal_chunk == EMPTY_CHUNK:
      continue
    print("Rebuttal chunk")
    print(" ".join(rebuttal_chunk))
    print('<table class="table">')
    fake_j = 0
    for j, review_chunk in enumerate(example["review_text"]):
      if review_chunk == EMPTY_CHUNK:
        continue
      else:
        fake_j += 1
      print("<tr><td>",vote_map[i][fake_j]/(4 * len(MODEL_MAP)), "</td><td>",
      " ".join(review_chunk), "</td> <td>", ",".join(voter_map[i][fake_j]),
      "</td></tr>")
    print("</table>")
  

def summarize_results(prediction_map, datasets):
  for review_sid, example in datasets["train"].items():
    vote_map = collections.defaultdict(lambda:collections.defaultdict(int))
    voter_map = collections.defaultdict(lambda:collections.defaultdict(list))
    for model, predictions in prediction_map.items():
      for k, v in predictions.items():
        print("----------", k, v)
      for rebuttal_chunk_i, rebuttal_chunk_predictions in enumerate(predictions[review_sid]):
        print("@@@@", rebuttal_chunk_predictions)
        for multiplier, idx in zip([4,2,1], rebuttal_chunk_predictions):
          vote_map[rebuttal_chunk_i][idx] += multiplier
          voter_map[rebuttal_chunk_i][idx].append(model)
    print("Review:", review_sid)
    print_with_votes(example, vote_map, voter_map)
          
def main():

  args = parser.parse_args()
  
  prediction_map = {
      "sentence": {},
      "chunk": {}
      }
  for discourse_unit in utils.DiscourseUnit.ALL:
    datasets = load_dataset_splits(args.data_dir, discourse_unit) 
    prediction_map = {}
    for model_name, model_bla in MODEL_MAP.items():
      model = model_bla(datasets)
      predictions = model.predict()
      prediction_map[model_name] = predictions

    summarize_results(prediction_map, datasets)


if __name__ == "__main__":
  main()
