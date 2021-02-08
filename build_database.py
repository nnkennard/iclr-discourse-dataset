import collections
import corenlp
import json
import openreview
import random
import sys

import openreview_lib as orl

random.seed(47)

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

def get_unstructured(conference, guest_client, output_dir):
  forums =  get_sampled_forums(conference, guest_client, 1).forums
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

def get_traindev(conference, guest_client, output_dir):
  forums = get_sampled_forums(conference, guest_client, 1).forums
  random.shuffle(forums)
  offsets = {
      orl.SubSplit.DEV :(0, int(0.2*len(forums))),
      orl.SubSplit.TRAIN : (int(0.2*len(forums)), int(0.8*len(forums))),
      orl.SubSplit.TEST : (int(0.8*len(forums)), int(1.1*len(forums)))
      }
  sub_split_forum_map = {
      sub_split: forums[start:end]
      for sub_split, (start, end) in offsets.items()
      }
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
      "abstracts": None
      }, f)

def get_truetest(conference, guest_client, output_dir):
  forums =  get_sampled_forums(conference, guest_client, 0.2).forums
  pair_texts = get_pair_text_from_forums(forums, guest_client)
  with open(output_dir + "/" + orl.Split.TRUETEST + ".json", 'w') as f:
    json.dump({
      "conference": conference,
      "split": orl.Split.TRUETEST,
      "subsplit": orl.SubSplit.TEST,
      "review_rebuttal_pairs": pair_texts,
      "abstracts": None
      }, f)


def main():
  guest_client = openreview.Client(baseurl='https://api.openreview.net')

  SPLIT_TO_CONFERENCE = {
    orl.Split.UNSTRUCTURED: orl.Conference.iclr18,
    orl.Split.TRAINDEV: orl.Conference.iclr19,
    orl.Split.TRUETEST: orl.Conference.iclr20
    }

  output_dir = "unlabeled/"

  get_unstructured(
      SPLIT_TO_CONFERENCE[orl.Split.UNSTRUCTURED], guest_client, output_dir)
  get_traindev(
      SPLIT_TO_CONFERENCE[orl.Split.TRAINDEV], guest_client, output_dir)
  get_truetest(
      SPLIT_TO_CONFERENCE[orl.Split.TRUETEST], guest_client, output_dir)


if __name__ == "__main__":
  main()
