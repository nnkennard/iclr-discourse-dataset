import argparse
import collections
import json
import openreview
import sys
import random

from tqdm import tqdm

#import openreview_lib as orl

class ConferenceName(object):
  iclr18 = "iclr18"
  iclr19 = "iclr19"
  iclr20 = "iclr20"
  ALL = [iclr18, iclr19, iclr20]

INVITATION_MAP = {
    ConferenceName.iclr18:'ICLR.cc/2018/Conference/-/Blind_Submission',
    ConferenceName.iclr19:'ICLR.cc/2019/Conference/-/Blind_Submission',
    ConferenceName.iclr20:'ICLR.cc/2020/Conference/-/Blind_Submission',
}

Conference = collections.namedtuple("Conference", "conference id_map url".split())

parser = argparse.ArgumentParser(
    description='Create stratified train/dev/test split of ICLR 2018 - 2020 forums.')
parser.add_argument('-o', '--outputdir', default="splits/",
    type=str, help="Where to dump output json file")

random.seed(23)

def get_forum_ids(guest_client, forum_id):
  return guest_client.get_notes(forum=forum_id)

TRAIN, DEV, TEST = ("train", "dev", "test")

def split_forums(forums):
  random.shuffle(forums)
  train_threshold = int(0.6 * len(forums))
  dev_threshold = int(0.8 * len(forums))

  return (forums[:train_threshold],
      forums[train_threshold:dev_threshold], forums[dev_threshold:])

class Forum(object):
  def __init__(self, forum_id, client):
    self.forum_id = forum_id
    notes = client.get_notes(forum=forum_id)
    self.num_notes = len(notes)


def get_all_conference_forums(conference, client):
  return list(openreview.tools.iterget_notes(
    client, invitation=INVITATION_MAP[conference]))


def get_stratified_forums(conference, client):
  forums = get_all_conference_forums(conference, client)
  forums_by_len = collections.defaultdict(list)
  forum_to_len_map = {}

  for forum in tqdm(forums):
    num_comments = len(get_forum_ids(client, forum.id))
    forums_by_len[num_comments].append(forum.id)
    forum_to_len_map[forum.id] = num_comments

  assert set(forum_to_len_map.keys()) == set(forum.id for forum in forums)

  total_num_forums = sum(len(forums) for forums in forums_by_len.values())
  bot_quintile_cumulative = int(0.2 * total_num_forums)
  top_quintile_cumulative = int(0.8 * total_num_forums)

  bottom_quintile_num_posts = None
  top_quintile_num_posts = None

  num_comments_seen = 0
  for num_comments in sorted(forums_by_len.keys()):
    num_new_comments = len(forums_by_len[num_comments])
    num_comments_seen += num_new_comments
    if bottom_quintile_num_posts is None:
      if num_comments_seen > bot_quintile_cumulative:
        bottom_quintile_num_posts = num_comments
    elif top_quintile_num_posts is None:
      if num_comments_seen > top_quintile_cumulative:
        top_quintile_num_posts = num_comments
        break

  small_forums = [forum 
      for forum, num_comments in forum_to_len_map.items()
      if num_comments <= bot_quintile_cumulative]

  large_forums = [forum 
      for forum, num_comments in forum_to_len_map.items()
      if num_comments >= top_quintile_cumulative]

  medium_forums = [forum 
    for forum in forum_to_len_map.keys()
    if forum not in small_forums and forum not in large_forums]

  forum_name_map = collections.defaultdict(list)

  for forum_set in [medium_forums, small_forums, large_forums]:
    train, dev, test = split_forums(forum_set)
    forum_name_map[TRAIN] += train
    forum_name_map[DEV] += dev
    forum_name_map[TEST] += test

  assert set(sum(forum_name_map.values(), [])) == set(forum_to_len_map.keys())

  clean = dict(forum_name_map)

  return Conference(conference, clean, INVITATION_MAP[conference])._asdict()


def get_unstructured_ids(conference, client):
  return get_sampled_forums(conference, client, 1)


def get_sampled_forums(conference, client, sample_rate):
  forums = [forum.id
            for forum in get_all_conference_forums(conference, client)]
  if sample_rate == 1:
    pass
  else:
    random.shuffle(forums)
    forums = forums[:int(sample_rate * len(forums))]
  return Conference(conference, forums, INVITATION_MAP[conference])._asdict()


TEST_SAMPLE_RATE = 0.1
def get_datasets(client):
  return {
      "unstructured": get_unstructured_ids(ConferenceName.iclr18, client),
      "traindev": get_stratified_forums(ConferenceName.iclr19, client),
      "truetest": get_sampled_forums(ConferenceName.iclr20, client,
        TEST_SAMPLE_RATE) 
      }


def main():

  args = parser.parse_args()

  guest_client = openreview.Client(baseurl='https://api.openreview.net')
  output_file = "".join([args.outputdir, "/iclr_discourse_dataset_split.json"])
  with open(output_file, 'w') as f:
    json.dump(get_datasets(guest_client), f)


if __name__ == "__main__":
  main()
