import argparse
import collections
import json
import sys
from termcolor import colored
import tqdm


import openreview_db as ordb

parser = argparse.ArgumentParser(
    description='Load OpenReview data from a sqlite3 database.')
parser.add_argument('-d', '--dbfile', default="../../db/or.db",
    type=str, help='path to database file')
parser.add_argument('-n', '--numexamples', default=-1,
    type=int, help='number of examples to dump as string. If -1, dump all')

def print_match(review_chunks, rebuttal_chunks, match):
  review_loc, rebuttal_loc, match_tokens = match
  _, review_chunk, review_tok = review_loc
  _, rebuttal_chunk, rebuttal_tok = rebuttal_loc

  print("Match tokens ({0}):".format(len(match_tokens)), " ".join(match_tokens))
  for i, chunk in enumerate(review_chunks):
    if i == review_chunk:
      tokens = sum(chunk, [])
      print(" ".join(tokens[: review_tok]))
      print(colored(" ".join(tokens[review_tok:review_tok + len(match_tokens)]),
        "red"))
      print(" ".join(tokens[review_tok + len(match_tokens):]))
    else:
      print(" ".join(" ".join(sentence) for sentence in chunk))



def print_match(chunks, location, match_tokens, color):
  _, match_chunk, tok = location

  print("Match tokens ({0}):".format(len(match_tokens)), " ".join(match_tokens))
  for i, chunk in enumerate(chunks):
    if i == match_chunk:
      tokens = sum(chunk, [])
      print(" ".join(tokens[: tok]))
      print(colored(" ".join(tokens[tok:tok + len(match_tokens)]),
        color))
      print(" ".join(tokens[tok + len(match_tokens):]))
    else:
      print(" ".join(" ".join(sentence) for sentence in chunk))

def main():

  match_map = collections.defaultdict(list)
  with open("out.json", 'r') as f:
    for i in json.load(f):
      match_map[i[1][0]].append(i)
  
  for k, v in match_map.items():
    print(k)
    for i in v:
      print(i)

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
    if rebuttal_sid in match_map:
      
      cur.execute("SELECT * FROM traindev WHERE sid=?",
          (review_sid,))
      review_chunks, = ordb.crunch_text_rows(cur.fetchall()).values()
      for chunk in review_chunks:
            cur.execute("SELECT * FROM traindev WHERE sid=?",
          (rebuttal_sid,))
      rebuttal_chunks, = ordb.crunch_text_rows(cur.fetchall()).values()
      match = match_map[rebuttal_sid][0]
      print_match(review_chunks, match[0], match[2], "red")
      print_match(rebuttal_chunks, match[1], match[2], "green")
      print("=" * 80)



if __name__ == "__main__":
  main()
