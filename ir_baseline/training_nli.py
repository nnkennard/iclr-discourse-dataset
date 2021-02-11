"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
import argparse
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import glob
import csv

parser = argparse.ArgumentParser(
    description='Build database of review-rebuttal pairs')
parser.add_argument('-i', '--inputdir',
    default="../review_rebuttal_pair_dataset/ws/",
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


def build_samples(train_dir):
  train_samples = []
  train_samples.append(InputExample(texts=["hi friend!", "hello friend!"], label=0))
  train_samples.append(InputExample(texts=["bye friend!", "see ya friend!"], label=1))
  return train_samples
  with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
      if row['split'] == 'train':
        label_id = label2int[row['label']]
        train_samples.append(InputExample(texts=[qd_1, qd_2], label=label_id))


def build_dataloader(input_dir, batch_size):
  samples = build_samples(input_dir)
  return DataLoader(samples, shuffle=True, batch_size=batch_size)


# Configure the training
num_epochs = 1


def main():

  args = parser.parse_args()
  model = build_model()

  train_loss = losses.SoftmaxLoss(
      model=model,
      sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
      num_labels=NUM_LABELS)

  dev_samples = build_samples(args.inputdir)
  dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
      dev_samples, batch_size=TRAIN_BATCH_SIZE, name='sts-dev')

  for epoch_i in range(num_epochs):
    for train_folder in glob.glob(args.inputdir + "/*"):
      train_loader = build_dataloader(train_folder, TRAIN_BATCH_SIZE)
      warmup_steps = math.ceil(len(train_loader) *
                               0.1)  #10% of train data for warm-up
      model.fit(train_objectives=[(train_loader, train_loss)],
                evaluator=dev_evaluator,
                epochs=1,
                evaluation_steps=1000,
                warmup_steps=warmup_steps,
                output_path=model_save_path)


if __name__ == "__main__":
  main()
