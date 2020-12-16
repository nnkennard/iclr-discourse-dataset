import argparse
import collections
import json
import sqlite3
import sys
import tqdm

import karp_rabin
import openreview_lib as orl
import openreview_db as ordb

parser = argparse.ArgumentParser(
    description='Load OpenReview data from a sqlite3 database.')
parser.add_argument('-d', '--dbfile', default="../../db/or.db",
    type=str, help='path to database file')
parser.add_argument('-n', '--numexamples', default=-1,
    type=int, help='number of examples to dump as string. If -1, dump all')

EMPTY_CHUNK = ["<br>"]


Location = collections.namedtuple("Location", "sid chunk token".split())

Match = Match = collections.namedtuple("Match",
    "review_location rebuttal_location tokens".split())


def get_longest_subsequence_match(review_chunk, rebuttal_chunk):
  review_tokens = sum(review_chunk, [])
  rebuttal_tokens = sum(rebuttal_chunk, [])
  if rebuttal_tokens == EMPTY_CHUNK or rebuttal_tokens == EMPTY_CHUNK:
    return []
  match_indices = karp_rabin.karp_rabin(review_tokens, rebuttal_tokens)
  return match_indices

def detect_subsequence_matches(review_chunks, rebuttal_chunks, review_sid,
    rebuttal_sid):
  matches = []
  for review_idx, review_chunk in enumerate(review_chunks):
    for rebuttal_idx, rebuttal_chunk in enumerate(rebuttal_chunks):
      mini_matches = get_longest_subsequence_match(review_chunk, rebuttal_chunk)
      if mini_matches:
        for mini_match in mini_matches:
          matches.append(Match(
            review_location=Location(review_sid, review_idx,
              mini_match.review_start),
            rebuttal_location=Location(rebuttal_sid, rebuttal_idx,
              mini_match.rebuttal_start), tokens=mini_match.tokens))

  return matches 

def get_json(obj):
    return json.loads(
            json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)))
              )

def main():

  args = parser.parse_args()
  conn = ordb.create_connection(args.dbfile)
  if conn is None:
    print("Error")
    return
  cur = conn.cursor()

  if args.numexamples == -1:
      get_rows_command = "SELECT * FROM traindev_pairs WHERE split=?"
  else:
      get_rows_command = ("SELECT * FROM traindev_pairs WHERE "
      "split=? LIMIT {0}").format(args.numexamples)

  matches = []
  cur.execute(get_rows_command, ("train",))
  rows = cur.fetchall()
  for row in tqdm.tqdm(list(rows)):
    review_sid = row["review_sid"]
    rebuttal_sid = row["rebuttal_sid"]
    cur.execute("SELECT * FROM traindev WHERE sid=?",
        (review_sid,))
    review_chunks, = ordb.crunch_text_rows(cur.fetchall()).values()
    for chunk in review_chunks:
          cur.execute("SELECT * FROM traindev WHERE sid=?",
        (rebuttal_sid,))
    rebuttal_chunks, = ordb.crunch_text_rows(cur.fetchall()).values()

    matches += detect_subsequence_matches(review_chunks, rebuttal_chunks,
        review_sid, rebuttal_sid)

  with open("out1.json", 'w') as f:
    json.dump([get_json(x) for x in matches], f)


if __name__ == "__main__":
  main()
