import argparse
import collections
import json

import utils
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


def get_pairs(cursor, dataset_name, split, numexamples):
  get_pairs_command = "SELECT * FROM {0}_pairs WHERE split=?".format(
      dataset_name)
  if not numexamples == -1:
    get_pairs_command += " LIMIT {0}".format(numexamples)

  return cursor.execute(get_pairs_command, (split,)).fetchall()


def get_text(cursor, dataset_name, sid, discourse_unit):
  rows = cursor.execute("SELECT * FROM {0} WHERE sid=?".format(dataset_name),
                (sid,))
  chunks, = ordb.crunch_text_rows(rows).values()
  if discourse_unit == utils.DiscourseUnit.sentence:
    text = sum(chunks, [])
  else:
    text = [sum(chunk, []) for chunk in chunks]
  return text 


def get_dataset(cursor, dataset_name, split, numexamples):
  print("Dataset name: ", dataset_name, "Split: ", split)

  dataset = utils.Dataset(dataset_name, split, [])
  pairs = get_pairs(cursor, dataset_name, split, numexamples)

  for pair in tqdm(pairs):
    review_sid = pair["review_sid"]
    rebuttal_sid = pair["rebuttal_sid"]

    review_text = get_text(cursor, dataset_name, review_sid,
            utils.DiscourseUnit.sentence)
    rebuttal_text = get_text(cursor, dataset_name, rebuttal_sid,
            utils.DiscourseUnit.chunk)

    dataset.examples.append(utils.Example(
    review_sid=review_sid,
    rebuttal_sid=rebuttal_sid,
    review_text=review_text,
    rebuttal_text=rebuttal_text,
    labels = [None] * len(rebuttal_text),
    ))

  return datasets

def main():

  args = parser.parse_args()

  cursor = ordb.get_cursor(args.dbfile)

  for dataset_name, split in utils.DATASET_NAMES:
    dataset = get_dataset(cursor, dataset_name, split, args.numexamples)
    filename = utils.get_dataset_filename(args.out_dir, dataset_name, split)
    with open(filename, 'w') as f:
        json.dump(utils.dump_dataset(dataset), f)


if __name__ == "__main__":
  main()
