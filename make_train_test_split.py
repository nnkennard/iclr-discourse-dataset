import argparse
import collections
import json
import openreview
import sys
import random

from tqdm import tqdm

import openreview_lib as orl

class ConferenceName(object):
  iclr18 = "iclr18"
  iclr19 = "iclr19"
  iclr20 = "iclr20"
  ALL = [iclr18, iclr19, iclr20]

INVITATION_MAP = {
    Conference.iclr18:'ICLR.cc/2018/Conference/-/Blind_Submission',
    Conference.iclr19:'ICLR.cc/2019/Conference/-/Blind_Submission',
    Conference.iclr20:'ICLR.cc/2020/Conference/-/Blind_Submission',
}

Conference = namedtuple("Conference", "conference id_map url".split())

parser = argparse.ArgumentParser(
    description='Create stratified train/dev/test split of ICLR 2018 - 2020 forums.')
parser.add_argument('-o', '--outputdir', default="../splits/",
    type=str, help="Where to dump output json file")

random.seed(23)

def get_forum_ids(guest_client, invitation):
  submissions = openreview.tools.iterget_notes(
        guest_client, invitation=invitation)
  return [n.forum for n in submissions]

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


def make_stratified_split(forum_ids, client):
  len_counter = collections.Counter() # Cumulative count of forum lengths

  forums = []
  for forum_id in tqdm(forum_ids):
    new_forum = Forum(forum_id, guest_client)
    len_counter[new_forum.num_notes] += 1
    forums.append(new_forum)

  # Ensuring that top and bottom quartile (and middle) are spread evenly between
  # splits
  total_notes = sum(len_counter.values())
  bottom_quintile_count = int(0.2 * total_notes)
  top_quintile_count = int(0.8 * total_notes)

  bottom_quintile_num_posts = None
  top_quintile_num_posts = None

  num_notes_seen = 0
  for num_posts in sorted(len_counter.keys()):
    num_notes = len_counter[num_posts]
    num_notes_seen += num_notes
    if bottom_quintile_num_posts is None:
      if num_notes_seen > bottom_quintile_count:
        bottom_quintile_num_posts = num_posts
    elif top_quintile_num_posts is None:
      if num_notes_seen > top_quintile_count:
        top_quintile_num_posts = num_posts
        break

  small_forums = [forum for forum in forums if forum.num_notes <=
    bottom_quintile_num_posts]

  large_forums = [forum for forum in forums if forum.num_notes >=
    top_quintile_num_posts]

  medium_forums = [forum 
    for forum in forums
    if forum not in small_forums and forum not in large_forums]

  forum_name_map = collections.defaultdict(list)

  for forum_set in [medium_forums, small_forums, large_forums]:
    train, dev, test = split_forums(forum_set)
    forum_name_map[TRAIN] += [forum.forum_id for forum in train]
    forum_name_map[DEV] += [forum.forum_id for forum in dev]
    forum_name_map[TEST] += [forum.forum_id for forum in test]

  clean = dict(forum_name_map)

  return Conference(conference, forum_name_map, INVITATION_MAP[conference])


def get_unstructured_ids(conference):
  return get_sampled_forums(conference, 1)


def get_sampled_forums(conference, sample_rate):
  forums = get_all_conference_forums(conference)
  if sample_rate == 1:
    pass
  else:
    random.shuffle(forums)
    forums = forums[:int(sample_rate * len(forums))]
  return Conference(conference, forums, INVITATION_MAP[conference])


def get_datasets():
  output = {
      "unstructured": get_unstructured_ids(orl.Conference.iclr18),
      "traindev": get_stratified_forums(orl.Conference.irate):
      forums = get_all_conference_forums("truetest": get_sampled_forums(orl.Conference.iclr20) 
      }


def main():

  args = parser.parse_args()

  guest_client = openreview.Client(baseurl='https://api.openreview.net')

  forum_ids = get_forum_ids(guest_client, orl.INVITATION_MAP[args.conference])

  datasets = get_datasets()
  output_file = "".join([args.outputdir, "/", args.conference, "_split.json"])
  with open(output_file, 'w') as f:
    f.write(json.dumps(dataset))


if __name__ == "__main__":
  main()
