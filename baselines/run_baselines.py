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
parser.add_argument('-b', '--dbfile', default="../db/or.db",
        type=str, help="Database in sqlite3 format")
parser.add_argument('-d', '--data_dir', default="datasets/",
        type=str, help="Dataset directory")
parser.add_argument('-r', '--result_dir', default="results/",
        type=str, help="Results directory")
parser.add_argument('-n', '--numexamples', default=-1,
        type=int, help="Number of examples per dataset; -1 to include all")


MODEL_MAP = {
  "tfidf": models.TfIdfModel,
  "sbert": models.SentenceBERTModel,
  #"roberta": models.RobertaModel,
  "bm25": models.BMModel,
    }

def load_dataset_splits(data_dir, discourse_unit):
  datasets = {}
  for split in ["train", "dev", "test"]:
    with open(utils.get_dataset_filename(data_dir, "traindev", split,
      discourse_unit), 'r') as f:
      obj = json.load(f)
      datasets[split] = {example["review_sid"]: example
          for example in obj["examples"]}
  return datasets 

def main():

  args = parser.parse_args()
  
  for discourse_unit in utils.DiscourseUnit.ALL:
    for model_name, model_bla in MODEL_MAP.items():
      datasets = load_dataset_splits(args.data_dir, discourse_unit) 
      model = model_bla(datasets)
      predictions = model.predict()
      print(predictions)

  exit()

  from transformers import RobertaTokenizer, RobertaForQuestionAnswering
  import torch

  tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  model = RobertaForQuestionAnswering.from_pretrained('roberta-base')

  question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
  inputs = tokenizer(question, text, return_tensors='pt')
  start_positions = torch.tensor([1])
  end_positions = torch.tensor([3])

  outputs = model(**inputs, start_positions=start_positions,
      end_positions=end_positions)
  print(outputs)
  loss = outputs.loss
  start_scores = outputs.start_logits
  end_scores = outputs.end_logits


if __name__ == "__main__":
  main()
