import argparse
import collections
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from BasicEvaluator  import BasicEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import glob
import csv
import pickle

parser = argparse.ArgumentParser(
    description='Build database of review-rebuttal pairs')
parser.add_argument('-i', '--inputdir',
    default="../review_rebuttal_pair_dataset_debug/ws/",
    type=str, help='path to database file')
parser.add_argument('-d', '--debug', action='store_true',
                    help='truncate to small subset of data for debugging')



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

# Read the dataset
TRAIN_BATCH_SIZE = 16
NUM_LABELS = 2

model_save_path = 'output/training_nli_' + model_name.replace(
    "/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def build_model():

  # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
  word_embedding_model = models.Transformer(model_name)

  # Apply mean pooling to get one fixed sized sentence vector
  pooling_model = models.Pooling(
      word_embedding_model.get_word_embedding_dimension(),
      pooling_mode_mean_tokens=True,
      pooling_mode_cls_token=False,
      pooling_mode_max_tokens=False)
  model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
  return model

def get_stripped_lines(filename):
  with open(filename, 'r') as f:
    return [l.strip() for l in f.readlines()]

def get_this_review_lines(rebuttal_idx, keys, review_lines):
  reb_split, reb_example, reb_line = keys["query_keys"][rebuttal_idx]
  this_review_indices = []
  for rev_i, (split, example, _) in enumerate(keys["corpus_keys"]):
    if split == reb_split and example == reb_example:
      this_review_indices.append(rev_i)

def build_samples(train_dir, dataset, example_index, scores):
  examples_dir = "/".join([train_dir, dataset, str(example_index), ""])
  train_samples = []
  review_lines = get_stripped_lines(examples_dir + "review.txt")
  rebuttal_lines = get_stripped_lines(examples_dir + "rebuttal.txt")

  reb_scores = scores[(dataset, example_index)]
  for reb_i, sent_scores in reb_scores.items():
    for rev_i, score_i in enumerate(sent_scores):
      for rev_j, score_j in enumerate(sent_scores):
        if score_i > score_j:
          label = 0
        else:
          label = 1
        text_1 = review_lines[rev_i] + " BREAK " + rebuttal_lines[reb_i]
        text_2 = review_lines[rev_j] + " BREAK " + rebuttal_lines[reb_i]
        train_samples.append(InputExample(texts=[text_1, text_2], label=label))
        if len(train_samples) > 100:
          return train_samples

  return train_samples


def build_dataloader(input_dir, dataset, example_index, scores, batch_size):
  samples = build_samples(input_dir, dataset, example_index, scores)
  return DataLoader(samples, shuffle=True, batch_size=batch_size)


# Configure the training
num_epochs = 15


def main():

  args = parser.parse_args()
  model = build_model()

  train_loss = losses.SoftmaxLoss(
      model=model,
      sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
      num_labels=NUM_LABELS)

  score_map = collections.defaultdict(lambda : collections.defaultdict())
  with open(args.inputdir + "/scores.pickle", 'rb') as f:
    scores = pickle.load(f)

  for key, score_list in scores.items():
    dataset, pair_index = key
    if 'test' in dataset:
      continue
    for rev_i, score in enumerate(score_list):
      score_map[key][rev_i] = score

  dev_samples = sum([
    build_samples(args.inputdir, "traindev_dev", i, score_map)
    for i in range(6)
    ], [])
  dev_evaluator = BasicEvaluator.from_input_examples(
      dev_samples, model, batch_size=TRAIN_BATCH_SIZE, name='sts-dev')

  for epoch_i in range(num_epochs):
    num_examples = len(glob.glob(args.inputdir +"/traindev_train/*")) - 2
    num_examples = 20
    for example_i in range(num_examples):
      train_loader = build_dataloader(args.inputdir, "traindev_train",
          example_i, score_map, TRAIN_BATCH_SIZE)
      warmup_steps = math.ceil(len(train_loader) *
                               0.1)  #10% of train data for warm-up
      model.fit(train_objectives=[(train_loader, train_loss)],
                evaluator=dev_evaluator,
                epochs=1,
                evaluation_steps=1000,
                warmup_steps=warmup_steps,
                output_path=model_save_path)
      for input_ids, labels in train_loader:
        print(train_loss(input_ids, None))
        #output = model(input_ids)
        #print(output)


if __name__ == "__main__":
  main()
