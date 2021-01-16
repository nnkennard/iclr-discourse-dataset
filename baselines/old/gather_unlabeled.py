import argparse
import collections
import json
from nltk.corpus import stopwords
import sys
import tqdm
import re

import openreview_db as ordb
import openreview_lib as orl

parser = argparse.ArgumentParser(
    description='Load OpenReview data from a sqlite3 database.')
parser.add_argument('-d', '--dbfile', default="../db/or.db",
    type=str, help='path to database file')
parser.add_argument('-n', '--numexamples', default=-1,
    type=int, help='number of examples to dump as string. If -1, dump all')


NEWLINE_TOKEN = "<br>"
STOP_WORDS = set(stopwords.words('english')).union({",", ".", NEWLINE_TOKEN})
EMPTY_CHUNK = [NEWLINE_TOKEN]



def get_match_list(cur, review_sid, rebuttal_sid, table_name):
  review_tokens = chunks_to_tokens(ordb.crunch_note_text_rows(
      cur, review_sid, "traindev"))
  rebuttal_tokens = chunks_to_tokens(ordb.crunch_note_text_rows(
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

  get_pairs_command = "SELECT * FROM unstructured_pairs"

  cur.execute(get_pairs_command)
  pairs = cur.fetchall()

  text = []

  for pair in tqdm.tqdm(list(pairs)):
    review_tokens = orl.chunks_to_tokens(ordb.crunch_note_text_rows(
        cur, pair["review_sid"], "unstructured"))
    rebuttal_tokens = orl.chunks_to_tokens(ordb.crunch_note_text_rows(
        cur, pair["rebuttal_sid"], "unstructured"))
    text.append(review_tokens)
    text.append(rebuttal_tokens)

  filename = "unstructured.json"

  with open(filename, 'w') as f:
    json.dump({"text": text}, f)
  

if __name__ == "__main__":
  main()
