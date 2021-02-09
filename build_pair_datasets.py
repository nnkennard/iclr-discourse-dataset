import argparse
import collections
import corenlp
import json
import openreview
import os
import sys

import openreview_lib as orl

parser = argparse.ArgumentParser(
    description='Build database of review-rebuttal pairs')
parser.add_argument('-o', '--outputdir', default="review_rebuttal_pair_dataset/",
    type=str, help='path to database file')
parser.add_argument('-d', '--debug', action='store_true',
                    help='truncate to small subset of data for debugging')


CORENLP_ANNOTATORS = "ssplit tokenize"

def get_pair_text_from_forums(forums, guest_client):
  print("Getting pair text from ", len(forums), " forums")
  sid_map, pairs = orl.get_review_rebuttal_pairs(
      forums, guest_client)
  with corenlp.CoreNLPClient(
      annotators=CORENLP_ANNOTATORS, output_format='conll') as corenlp_client:
    return orl.get_pair_text(pairs, sid_map, corenlp_client)

def get_abstracts_from_forums(forums, guest_client):
  print("Getting abstracts")
  with corenlp.CoreNLPClient(
      annotators=CORENLP_ANNOTATORS, output_format='conll') as corenlp_client:
    return orl.get_abstract_texts(forums, guest_client, corenlp_client)

def get_unstructured(conference, guest_client, output_dir, sample_frac):
  """ Get unstructured data for domain pretraining.

      This includes review and rebuttal text (undifferentiated) and abstracts.

     Args:
        conference: Conference name (from openreview_lib.Conference)
        guest_client: OpenReview API guest client
        output_dir: Directory (which already exists) in which to dump jsons.
        sample_frac: Percentage of examples to retain

      Returns:
        Nothing -- just dumps to json.
  """
  forums =  orl.get_sampled_forums(conference, guest_client, sample_frac).forums
  pair_texts = get_pair_text_from_forums(forums, guest_client)
  unstructured_text = []
  for pair in pair_texts:
    unstructured_text.append(pair["review_text"])
    unstructured_text.append(pair["rebuttal_text"])
  abstracts = get_abstracts_from_forums(forums, guest_client)
  with open(output_dir + "/" + orl.Split.UNSTRUCTURED + ".json", 'w') as f:
    json.dump({
      "conference": conference,
      "split": orl.Split.UNSTRUCTURED,
      "subsplit": orl.SubSplit.TRAIN,
      "review_rebuttal_text": unstructured_text,
      "abstracts": abstracts,
      }, f)

def get_traindev(conference, guest_client, output_dir, sample_frac):
  """ Get review-rebuttal pairs in train/dev/test split.

      Args:
        conference: Conference name (from openreview_lib.Conference)
        guest_client: OpenReview API guest client
        output dir: Directory (which already exists) in which to dump jsons.
        sample_frac: Percentage of examples to retain

      Returns:
        Nothing -- just dumps to json.
  """

  sub_split_forum_map = orl.get_sub_split_forum_map(conference, guest_client,
          sample_frac)

  for sub_split, sub_split_forums in sub_split_forum_map.items():
    pair_texts = get_pair_text_from_forums(sub_split_forums, guest_client)
    with open(
        "".join([output_dir, "/", orl.Split.TRAINDEV, "_",
          sub_split, ".json"]), 'w') as f:
      json.dump({
      "conference": conference,
      "split": orl.Split.TRAINDEV,
      "subsplit": sub_split,
      "review_rebuttal_pairs": pair_texts,
      }, f)

def get_truetest(conference, guest_client, output_dir, sample_frac):
  """ Get review-rebuttal pairs in sampled at 20% for test set.

      Args:
        conference: Conference name (from openreview_lib.Conference)
        guest_client: OpenReview API guest client
        output dir: Directory (which already exists) in which to dump jsons.
        sample_frac: Percentage of examples to retain

      Returns:
        Nothing -- just dumps to json.
  """
  forums =  orl.get_sampled_forums(conference, guest_client, sample_frac).forums
  pair_texts = get_pair_text_from_forums(forums, guest_client)
  with open(output_dir + "/" + orl.Split.TRUETEST + ".json", 'w') as f:
    json.dump({
      "conference": conference,
      "split": orl.Split.TRUETEST,
      "subsplit": orl.SubSplit.TEST,
      "review_rebuttal_pairs": pair_texts,
      }, f)


TRUETEST_SUBSAMPLE = 0.2

def main():

  args = parser.parse_args()
  if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

  guest_client = openreview.Client(baseurl='https://api.openreview.net')

  # There are three splits:
  #   UNSTRUCTURED: for unstructured pretraining (like Don't Stop Pretraining
  #   paper) (ICLLR 2018)
  #   TRUETEST: totally unseen test set (ICLR 2020)
  #   TRAINDEV: normal train/dev/test split (ICLR 2019)

  SPLIT_TO_CONFERENCE = {
    orl.Split.UNSTRUCTURED: orl.Conference.iclr18,
    orl.Split.TRAINDEV: orl.Conference.iclr19,
    orl.Split.TRUETEST: orl.Conference.iclr20
    }

  if args.debug:
    base_sample_frac = 0.01
  else:
    base_sample_frac = 1.0

  get_unstructured(
      SPLIT_TO_CONFERENCE[orl.Split.UNSTRUCTURED], guest_client,
          args.outputdir, base_sample_frac)
  get_traindev(
      SPLIT_TO_CONFERENCE[orl.Split.TRAINDEV], guest_client, args.outputdir,
      base_sample_frac)
  get_truetest(
      SPLIT_TO_CONFERENCE[orl.Split.TRUETEST], guest_client, args.outputdir,
      base_sample_frac * TRUETEST_SUBSAMPLE)


if __name__ == "__main__":
  main()
