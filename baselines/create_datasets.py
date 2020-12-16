import argparse
import collections
import json

import openreview_db as ordb

from tqdm import tqdm



parser = argparse.ArgumentParser(
        description='Create datasets for training baseline models')
parser.add_argument('-d', '--dbfile', default="../db/or.db",
        type=str, help="Database in sqlite3 format")
parser.add_argument('-o', '--out_dir', default="datasets/",
        type=str, help="Dataset directory")
parser.add_argument('-n', '--numexamples', default=-1,
        type=int, help="Number of examples per dataset; -1 to include all")

Dataset = collections.namedtuple("Dataset",
  "dataset_name split discourse_unit examples".split())
Example = collections.namedtuple("Example",
  "review_sid rebuttal_sid review_text rebuttal_text labels")


class DiscourseUnit(object):
  sentence = "sentence"
  chunk = "chunk"
  ALL = [sentence, chunk]

DATASET_NAMES = [
    ("traindev", "train"),
    ("traindev", "dev"),
    ("traindev", "test"),
    ("truetest", "test")
    ]


def get_pairs(cursor, dataset_name, split, numexamples):
  get_pairs_command = "SELECT * FROM {0}_pairs WHERE split=?".format(
      dataset_name)
  if not numexamples == -1:
    get_pairs_command += " LIMIT {0}".format(numexamples)

  return cursor.execute(get_pairs_command, (split,)).fetchall()


def get_text(cursor, dataset_name, sid):
  rows = cursor.execute("SELECT * FROM {0} WHERE sid=?".format(dataset_name),
                (sid,))
  chunks, = ordb.crunch_text_rows(rows).values()
  sentences = sum(chunks, [])
  return {
      DiscourseUnit.chunk: chunks,
      DiscourseUnit.sentence: sentences
      }


def get_datasets(cursor, dataset_name, split, numexamples):
  numexamples = 10

  datasets = {
      discourse_unit:Dataset(dataset_name, split, discourse_unit, []) for
      discourse_unit in DiscourseUnit.ALL}

  pairs = get_pairs(cursor, dataset_name, split, numexamples)
  for pair in pairs:
    review_sid = pair["review_sid"]
    rebuttal_sid = pair["rebuttal_sid"]

    review_text = get_text(cursor, dataset_name, review_sid)
    rebuttal_text = get_text(cursor, dataset_name, rebuttal_sid)

    for discourse_unit in DiscourseUnit.ALL:
      datasets[discourse_unit].examples.append(Example(
        review_sid=review_sid,
        rebuttal_sid=rebuttal_sid,
        review_text=review_text[discourse_unit],
        rebuttal_text=rebuttal_text[discourse_unit],
        labels = [None] * len(rebuttal_text[discourse_unit]),
        ))
    
  return datasets

# TODO: Move to utils
def get_json(obj):
  return json.loads(
      json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o))))

def main():

  args = parser.parse_args()

  cursor = ordb.get_cursor(args.dbfile)

  for dataset_name, split in DATASET_NAMES:
    datasets = get_datasets(cursor, dataset_name, split, args.numexamples)
    for discourse_unit in DiscourseUnit.ALL:
      with open(args.out_dir + "/" + "_".join([dataset_name, split,
        discourse_unit]) + ".json", 'w') as f:
        json.dump(get_json(datasets[discourse_unit]), f)


if __name__ == "__main__":
  main()
