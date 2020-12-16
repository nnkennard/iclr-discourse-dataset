import collections
import json

class DiscourseUnit(object):
  sentence = "sentence"
  chunk = "chunk"
  ALL = [sentence, chunk]

Dataset = collections.namedtuple("Dataset",
  "dataset_name split discourse_unit examples".split())
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
