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

parser = argparse.ArgumentParser(
    description='Load OpenReview data from a sqlite3 database.')
parser.add_argument('-d', '--dbfile', default="../../db/or.db",
    type=str, help='path to database file')
parser.add_argument('-n', '--numexamples', default=-1,
    type=int, help='number of examples to dump as string. If -1, dump all')

NEWLINE_TOKEN = "<br>"
STOP_WORDS = set(stopwords.words('english')).union({",", ".", NEWLINE_TOKEN})
EMPTY_CHUNK = [NEWLINE_TOKEN]
Location = collections.namedtuple("Location",
  "sid piece_type piece_idx token_idx".split())
Match = Match = collections.namedtuple("Match",
    "review_location rebuttal_location tokens".split())

class MatchSet(object):
  def __init__(self, cur, review_sid, rebuttal_sid, table_name):
    review_tokens = sum(sum(ordb.crunch_note_text_rows(
        cur, review_sid, "traindev"), []), [])
    rebuttal_tokens = sum(sum(ordb.crunch_note_text_rows(
        cur, rebuttal_sid, "traindev"), []), [])

    match_indices = karp_rabin.karp_rabin(review_tokens, rebuttal_tokens)
    for i in sorted(match_indices, key=lambda x:x.tokens):
      if set(t.lower() for t in i.tokens).issubset(STOP_WORDS):
        continue
      print(i)

    print(len(match_indices))

    self.matches = [match._asdict() 
        for match in list(
          reversed(sorted(matches, key=lambda x:len(x.tokens))))]
  

  def _get_longest_subsequence_match(self, review_sentence, rebuttal_sentence):
    if rebuttal_sentence == EMPTY_CHUNK or rebuttal_sentence == EMPTY_CHUNK:
      return []
    match_indices = karp_rabin.karp_rabin(review_sentence, rebuttal_sentence)
    print(match_indices)
    return match_indices


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
    matches[pair["review_sid"]] = MatchSet(cur, pair["review_sid"], pair["rebuttal_sid"],
        "traindev")

  filename = "".join(["matches_traindev_", str(len(pairs)), ".pkl"])

  with open(filename, 'w') as f:
    pickle.dump(
  

if __name__ == "__main__":
  main()
