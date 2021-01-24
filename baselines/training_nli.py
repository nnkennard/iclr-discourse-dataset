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

import new_softmax_loss

from tqdm import tqdm

def prepare():
  logging.basicConfig(format='%(asctime)s - %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S',
                      level=logging.INFO,
                      handlers=[LoggingHandler()])

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
  train_loss = new_softmax_loss.SoftmaxLoss(model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=num_labels, num_vectors=3)
  return model, train_loss


def read_dataset(dataset_path, split):
  samples = []
  with open(dataset_path, 'r') as fIn:
    for line in tqdm(fIn):
      line_split, query, doc1, doc2, label = line.strip().split('\t')
      if line_split == split:
        samples.append(InputExample(texts=[query, doc1, doc2],
          label=int(label)))


def get_train_dataloader(nli_dataset_path, train_batch_size):
  logging.info("Read AllNLI train dataset")
  train_samples = read_dataset(nli_dataset_path, "train")
  return DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


def build_dev_evaluator(sts_dataset_path, train_batch_size):
  logging.info("Read STSbenchmark dev dataset")
  dev_samples = read_dataset(sts_dataset_path, "dev")
  return SameOldEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')


def test(model_save_path, sts_dataset_path, train_batch_size):
  test_samples = read_dataset(sts_dataset_path, "test")
  model = SentenceTransformer(model_save_path)
  test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
              test_samples, batch_size=train_batch_size, name='sts-test')
  test_evaluator(model, output_path=model_save_path)


def main():
  dataset_path = '../test_unlabeled/examples.tsv'
  prepare()
  model, train_loss = build_model(num_labels=2)

  train_batch_size = 16
  train_dataloader = get_train_dataloader(dataset_path, train_batch_size)
  dev_evaluator = build_dev_evaluator(dataset_path, train_batch_size)

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


if __name__ == "__main__":
  main()
