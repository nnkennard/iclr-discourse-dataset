import collections
import json
import math

class DiscourseUnit(object):
  sentence = "sentence"
  chunk = "chunk"
  ALL = [sentence, chunk]

Dataset = collections.namedtuple("Dataset",
  "dataset_name split examples".split())
Example = collections.namedtuple("Example",
  "review_sid rebuttal_sid review_text rebuttal_text labels")
Prediction = collections.namedtuple("Prediction",
  "review_sid rebuttal_sid labels")

DATASET_NAMES = [
    ("traindev", "train"),
    ("traindev", "dev"),
    ("traindev", "test"),
    ("truetest", "test")
    ]

def get_dataset_filename(dataset_dir, dataset_name, split, discourse_unit):
  return dataset_dir + "/" + "_".join([dataset_name, split,
        discourse_unit]) + ".json"

def dump_dataset(dataset):
  dataset_2 = Dataset(dataset.dataset_name, dataset.split,
      dataset.discourse_unit,
      [example._asdict() for example in dataset.examples])
  return dataset_2._asdict()


def sparse_cosine_denominator(vec):
  return math.sqrt(sum(math.pow(x, 2) for x in vec.values()))

def sparse_cosine(vec1, vec2):
  dvec1 = dict(vec1)
  dvec2 = dict(vec2)

  if not dvec1.keys() or not dvec2.keys():
    return 0.0

  num_acc = 0.0
  for k, v in dvec1.items():
    if k in dvec2:
      num_acc += v * dvec2[k]

  return num_acc / (sparse_cosine_denominator(
    dvec1) * sparse_cosine_denominator(dvec2))


