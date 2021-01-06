import collections
import corenlp
import json
import openreview
import random
import sys

import new_openreview_lib as norl

CORENLP_ANNOTATORS = "ssplit tokenize"
ForumList = collections.namedtuple("ForumList",
                                   "conference forums url".split())

def get_sampled_forums(conference, client, sample_rate):
  forums = [forum.id
            for forum in get_all_conference_forums(conference, client)]
  if sample_rate == 1:
    pass
  else:
    random.shuffle(forums)
    forums = forums[:int(sample_rate * len(forums))]
  return ForumList(conference, forums, norl.INVITATION_MAP[conference])

def get_all_conference_forums(conference, client):
  return list(openreview.tools.iterget_notes(
    client, invitation=norl.INVITATION_MAP[conference]))

def get_pair_text_from_forums(forums, guest_client):
  sid_map, pairs = norl.get_review_rebuttal_pairs(
      forums, guest_client)
  with corenlp.CoreNLPClient(
      annotators=CORENLP_ANNOTATORS, output_format='conll') as corenlp_client:
    return norl.get_pair_text(pairs, sid_map, corenlp_client)

def get_abstracts_from_forums(forums, guest_client):
  with corenlp.CoreNLPClient(
      annotators=CORENLP_ANNOTATORS, output_format='conll') as corenlp_client:
    return norl.get_abstract_texts(forums, guest_client, corenlp_client)

def get_unstructured(conference, guest_client, output_dir):
  forums =  get_sampled_forums(conference, guest_client, 1).forums
  pair_texts = get_pair_text_from_forums(forums, guest_client)
  unstructured_text = []
  for pair in pair_texts:
    unstructured_text.append(pair["review_text"])
    unstructured_text.append(sum(pair["rebuttal_text"], []))
  with open(output_dir + "/" + norl.Split.UNSTRUCTURED + ".json", 'w') as f:
    json.dump({
      "conference": conference,
      "split": norl.Split.UNSTRUCTURED,
      "subsplit": norl.SubSplit.TRAIN,
      "review_rebuttal_text": unstructured_text,
      "abstracts": get_abstracts_from_forums(forums, guest_client)
      }, f)

def get_traindev(conference, guest_client, output_dir):
  forums = get_sampled_forums(conference, guest_client, 1).forums
  random.shuffle(forums)
  offsets = {
      norl.SubSplit.DEV :(0, int(0.2*len(forums))),
      norl.SubSplit.TRAIN : (int(0.2*len(forums)), int(0.8*len(forums))),
      norl.SubSplit.TEST : (int(0.8*len(forums)), int(1.1*len(forums)))
      }
  sub_split_forum_map = {
      sub_split: forums[start:end]
      for sub_split, (start, end) in offsets.items()
      }
  for sub_split, sub_split_forums in sub_split_forum_map.items():
    pair_texts = get_pair_text_from_forums(sub_split_forums, guest_client)
    with open(
        "".join([output_dir, "/", norl.Split.TRAINDEV, "_",
          sub_split, ".json"]), 'w') as f:
      json.dump({
      "conference": conference,
      "split": norl.Split.TRAINDEV,
      "subsplit": sub_split,
      "review_rebuttal_pairs": pair_texts,
      "abstracts": None
      }, f)

def get_truetest(conference, guest_client, output_dir):
  forums =  get_sampled_forums(conference, guest_client, 0.2).forums
  pair_texts = get_pair_text_from_forums(forums, guest_client)
  with open(output_dir + "/" + norl.Split.TRUETEST + ".json", 'w') as f:
    json.dump({
      "conference": conference,
      "split": norl.Split.UNSTRUCTURED,
      "subsplit": norl.SubSplit.TEST,
      "review_rebuttal_pairs": pair_texts,
      "abstracts": None
      }, f)


def main():
  guest_client = openreview.Client(baseurl='https://api.openreview.net')

  SPLIT_TO_CONFERENCE = {
    norl.Split.UNSTRUCTURED: norl.Conference.iclr18,
    norl.Split.TRAINDEV: norl.Conference.iclr19,
    norl.Split.TRUETEST: norl.Conference.iclr20
    }

  output_dir = "unlabeled/"

  get_unstructured(
      SPLIT_TO_CONFERENCE[norl.Split.UNSTRUCTURED], guest_client, output_dir)
  get_traindev(
      SPLIT_TO_CONFERENCE[norl.Split.TRAINDEV], guest_client, output_dir)
  get_truetest(
      SPLIT_TO_CONFERENCE[norl.Split.TRUETEST], guest_client, output_dir)


if __name__ == "__main__":
  main()
