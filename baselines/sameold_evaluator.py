from . import SentenceEvaluator, SimilarityFunction
import torch
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
from ..readers import InputExample

class SameOldEvaluator(SentenceEvaluator):
    """
    Just normally evaluate the model

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self,
        queries : List[str], docs1: List[str], docs2: List[str], labels: List[int],
        loss, batch_size: int = 16,
        name: str = '', show_progress_bar: bool = False):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.
        """
        self.queries = queries
        self.docs1 = docs1
        self.docs2 = docs2
        self.labels = labels

        self.loss = loss


        assert len(set([
          len(self.queries), len(self.docs1), len(self.docs2), len(self.labels)
          ])) == 1

        self.batch_size = batch_size
        self.csv_file = "evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "f1"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        queries = []
        docs1 = []
        docs2 = []
        labels = []

        for example in examples:
            queries.append(example.texts[0])
            docs1.append(example.texts[1])
            docs2.append(example.texts[1])
            labels.append(example.label)
        return cls(queries, docs1, docs2, labels, **kwargs)

    def _my_encode(self, model, texts):
      return model.encode(
         texts, batch_size=self.batch_size,
         show_progress_bar=self.show_progress_bar, convert_to_numpy=True)



    def __call__(self,
        model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        embeddings1 = self._my_encode(model, self.queries)
        embeddings2 = self._my_encode(model, self.docs1)
        embeddings3 = self._my_encode(model, self.docs2)
        labels = self.scores

        loss = self.loss.forward([embeddings1, embeddings2, embeddings3], labels)

        return 5.0         # Need to return F1

        if self.main_similarity == SimilarityFunction.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
        else:
            raise ValueError("Unknown main_similarity value")
