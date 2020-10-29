import argparse
import json
import sqlite3
import sys
import tqdm

import lib.openreview_db as ordb

parser = argparse.ArgumentParser(
    description='Load OpenReview data from a sqlite3 database.')
parser.add_argument('-d', '--dbfile', default="db/or.db",
    type=str, help='path to database file')
parser.add_argument('-o', '--outputfile', default="django_output.json",
    type=str, help='path to django output file')

RELEVANT_FIELDS = "sid chunk_idx sentence_idx token_idx token".split()

def filter_token_row(token_row):
  return {k:v for k,v in token_row.items() if k in RELEVANT_FIELDS}

def get_token_rows(cur, sid):
  cur.execute("SELECT * FROM traindev WHERE sid=?", (sid,))
  return [filter_token_row(row) for row in cur.fetchall()]


def main():

  args = parser.parse_args()

  output_obj = {"meta":[], "tokens":[]}

  conn = ordb.create_connection(args.dbfile)
  if conn is not None:
    cur = conn.cursor()
    cur.execute("SELECT * FROM traindev_pairs WHERE split=?", ("train",))
    rows = list(cur.fetchall())
    for row in tqdm.tqdm(rows):
      output_obj["meta"].append({"review":row["review_sid"],
                                 "rebuttal":row["rebuttal_sid"],
                                 "title": row["title"],
                                 "review_author": row["review_author"]})
      output_obj["tokens"] += get_token_rows(cur, row["review_sid"])
      output_obj["tokens"] += get_token_rows(cur, row["rebuttal_sid"])

  with open(args.outputfile, 'w') as f:
    json.dump(output_obj, f)

if __name__ == "__main__":
  main()
