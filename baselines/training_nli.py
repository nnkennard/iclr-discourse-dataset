"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""

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
import csv

def prepare(nli_dataset_path, sts_dataset_path):
  #### Just some code to print debug information to stdout
  logging.basicConfig(format='%(asctime)s - %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S',
                      level=logging.INFO,
                      handlers=[LoggingHandler()])
  #### /print debug information to stdout

  #Check if dataset exsist. If not, download and extract  it

  if not os.path.exists(nli_dataset_path):
      util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

  if not os.path.exists(sts_dataset_path):
      util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


def build_model(num_labels):
  model_name = 'bert-base-uncased'

  # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
  word_embedding_model = models.Transformer(model_name)

  # Apply mean pooling to get one fixed sized sentence vector
  pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                 pooling_mode_mean_tokens=True,
                                 pooling_mode_cls_token=False,
                                 pooling_mode_max_tokens=False)

  model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
  train_loss = losses.SoftmaxLoss(model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=num_labels)
  return model, train_loss


def read_dataset(dataset_path, split):
  label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
  samples = []
  with gzip.open(dataset_path, 'rt', encoding='utf8') as fIn:
      reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
      for row in reader:
          if row['split'] == split:
            if 'label' in row:
              label_id = label2int[row['label']]
            else:
              label_id = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
  return samples[:10]

def get_train_dataloader(nli_dataset_path, train_batch_size):
  logging.info("Read AllNLI train dataset")
  train_samples = read_dataset(nli_dataset_path, "train")
  return DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


def build_dev_evaluator(sts_dataset_path, train_batch_size):
  logging.info("Read STSbenchmark dev dataset")
  dev_samples = read_dataset(sts_dataset_path, "dev")
  return EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')


def test(model_save_path, sts_dataset_path, train_batch_size):
  test_samples = read_dataset(sts_dataset_path, "test")
  model = SentenceTransformer(model_save_path)
  test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
              test_samples, batch_size=train_batch_size, name='sts-test')
  test_evaluator(model, output_path=model_save_path)


def main():
  nli_dataset_path = 'datasets/AllNLI.tsv.gz'
  sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
  prepare(nli_dataset_path, sts_dataset_path)
  model, train_loss = build_model(num_labels=3)

  train_batch_size = 16
  train_dataloader = get_train_dataloader(nli_dataset_path, train_batch_size)
  dev_evaluator = build_dev_evaluator(sts_dataset_path, train_batch_size)

  # Training configuration
  num_epochs = 1
  model_save_path = 'output/training_nli_bert-base-uncased-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

  # Train the model
  model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path
            )

  test(model_save_path, sts_dataset_path, train_batch_size)


if __name__ == "__main__":
  main()
