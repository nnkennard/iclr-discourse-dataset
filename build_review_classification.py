import argparse
import collections
import corenlp
import json
import openreview
import os
import sys

import openreview_lib as orl

parser = argparse.ArgumentParser(
    description='Build database of reviews with categorical labels')
parser.add_argument('-o', '--outputdir',
    default="review_classification_dataset/",
    type=str, help='path to database file')


CORENLP_ANNOTATORS = "ssplit tokenize"

def get_examples_from_forums(forums, guest_client):
  """ Convert reviews from a list of forums into classification examples.
      
      Args:
        forums: A list of forum ids
        conference: Conference name (from openreview_lib.Conference)

      Returns:
        A list of openreview_lib.ClassificationExamples
  """
  sid_map, pairs = orl.get_review_rebuttal_pairs(
      forums, guest_client)
  with corenlp.CoreNLPClient(
      annotators=CORENLP_ANNOTATORS, output_format='conll') as corenlp_client:
    return orl.get_classification_examples(pairs, "review",
        sid_map, corenlp_client)

def build_review_classification_dataset(conference, guest_client, output_dir):
  """ Build a dataset of review texts with categorical labels.

      Args:
        conference: Conference name (from openreview_lib.Conference)
        guest_client: OpenReview API guest client
        output dir: Directory (which already exists) in which to dump jsons.

      Returns:
        Nothing -- just dumps to json.
  """
  # "sub-split" just refers to train-dev-test split, but distinguishes between
  # this and unstructured/traindev/truetest split.
  sub_split_forum_map = orl.get_sub_split_forum_map(conference, guest_client)
  for sub_split, sub_split_forums in sub_split_forum_map.items():
    examples = get_examples_from_forums(sub_split_forums, guest_client)
    with open(
        "".join([output_dir, "/", orl.Split.TRAINDEV, "_",
          sub_split, ".json"]), 'w') as f:
      json.dump({
      "conference": conference,
      "split": orl.Split.TRAINDEV,
      "subsplit": sub_split,
      "examples": examples,
      }, f)


def main():

  args = parser.parse_args()
  if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

  guest_client = openreview.Client(baseurl='https://api.openreview.net')
  # In the overall dataset, we have one year for unstructured training, one year
  # for totally unseen test set, and one year which we split into
  # train/dev/test. For these experiments, we will stick to the train/dev/test
  # one which is ICLR 2019.
  TRAINDEV_CONFERENCE = orl.Conference.iclr19
  output_dir = "review_classification/"

  build_review_classification_dataset(
      TRAINDEV_CONFERENCE, guest_client, args.outputdir)


if __name__ == "__main__":
  main()
