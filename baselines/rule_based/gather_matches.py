import argparse
import collections
import json
from nltk.corpus import stopwords
import sys
import tqdm
import re

import karp_rabin
import trie
import openreview_db as ordb
import openreview_lib as orl

parser = argparse.ArgumentParser(
    description='Load OpenReview data from a sqlite3 database.')
parser.add_argument('-d', '--dbfile', default="../../db/or.db",
    type=str, help='path to database file')
parser.add_argument('-n', '--numexamples', default=-1,
    type=int, help='number of examples to dump as string. If -1, dump all')


#NEWLINE_TOKEN = "<br>"
STOP_WORDS = set(stopwords.words('english')).union({",", ".", orl.NEWLINE_TOKEN})
EMPTY_CHUNK = [orl.NEWLINE_TOKEN]


def get_match_list(cur, review_sid, rebuttal_sid, table_name):
  review_tokens = orl.chunks_to_tokens(ordb.crunch_note_text_rows(
      cur, review_sid, "traindev"))
  rebuttal_tokens = orl.chunks_to_tokens(ordb.crunch_note_text_rows(
      cur, rebuttal_sid, "traindev"))

  match_indices = karp_rabin.karp_rabin(review_tokens, rebuttal_tokens)
  final_matches = []
  for match in sorted(match_indices, key=lambda x:x.tokens):
    if set(t.lower() for t in match.tokens).issubset(STOP_WORDS):
      continue
    final_matches.append(match)

  return [match._asdict() 
      for match in list(
        reversed(sorted(final_matches, key=lambda x:len(x.tokens))))]


def main():

  args = parser.parse_args()
  cur = ordb.get_cursor(args.dbfile)

  if args.numexamples == -1:
      get_pairs_command = "SELECT * FROM traindev_pairs WHERE split=?"
  else:
      get_pairs_command = ("SELECT * FROM traindev_pairs WHERE "
      "split=? LIMIT {0}").format(args.numexamples)

  cur.execute(get_pairs_command, ("train", ))
  pairs = cur.fetchall()

  matches = {}
  for pair in tqdm.tqdm(list(pairs)):
    matches[pair["review_sid"]] = get_match_list(
        cur, pair["review_sid"], pair["rebuttal_sid"], "traindev")

  filename = "".join(["matches_traindev.json"])

  with open(filename, 'w') as f:
    json.dump(matches, f)
  

if __name__ == "__main__":
  main()
