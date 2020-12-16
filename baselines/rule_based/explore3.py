import collections
import json
import sys
import tqdm
import re

from explore import Location, Match
import html_dump

import trie
import openreview_db as ordb

def get_match_map(match_filename):
  all_matches = collections.defaultdict(list)
  with open(match_filename, 'r') as f:
    matches = json.load(f)
    for match_data in matches:
      review_loc, rebuttal_loc, tokens = match_data
      match = Match(Location(*review_loc), Location(*rebuttal_loc), tokens)
      all_matches[match.review_location.sid].append(match)
  return all_matches


def highlight(chunks, location, length):
  highlighted_tokens = []
  for i, chunk in enumerate(chunks):
    tokens = sum(chunk, [])
    if i == location.chunk:
      new_chunk = []
      for i, token in enumerate(tokens):
        if i == location.token:
          new_chunk.append("<b>")
        new_chunk.append(token)
        if i+1 == location.token + length:
          new_chunk.append("</b>")
      highlighted_tokens += new_chunk
    else:
      highlighted_tokens += tokens
    highlighted_tokens.append("<br>")

  return " ".join(highlighted_tokens)

def longest_common_prefix(str1, str2):
  for i, char1 in enumerate(str1):
    if not char1 == str2[i]:
      return i

def clean_up_tokens(tokens):
  return re.sub('\d', '#', " ".join(tokens))

def find_prefixes(chunks):
  chunk_strs = []

  for i, chunk in enumerate(chunks):
    tokens = sum(chunk, [])
    chunk_strs.append(clean_up_tokens(tokens))

  chunk_trie = trie.Trie(chunk_strs)
  print(chunk_trie)
  
  print("*" * 80) 
    

def get_pairs(numexamples, cursor):
  if numexamples == -1:
      get_rows_command = "SELECT * FROM traindev_pairs WHERE split=?"
  else:
      get_rows_command = ("SELECT * FROM traindev_pairs WHERE "
      "split=? LIMIT {0}").format(numexamples)
  cursor.execute(get_rows_command, ("train",))
  return cursor.fetchall()


def rule_based_matcher(review_sid, rebuttal_sid, cursor, match_map):
  cursor.execute("SELECT * FROM traindev WHERE sid=?", (review_sid,))
  review_chunks, = ordb.crunch_text_rows(cursor.fetchall()).values()
  for chunk in review_chunks:
    cursor.execute("SELECT * FROM traindev WHERE sid=?",
        (rebuttal_sid,))
  rebuttal_chunks, = ordb.crunch_text_rows(cursor.fetchall()).values()
  relation_map = {i:[] for i in range(len(rebuttal_chunks))}
  for i, match in enumerate(match_map[review_sid]):
    chunk_tokens = sum(review_chunks[match.review_location.chunk], [])
    ratio = len(match.tokens) / len(chunk_tokens)
    print("\t".join([
      str(ratio),
      review_sid, str(match.review_location.chunk),
      rebuttal_sid, str(match.rebuttal_location.chunk),
      " ".join(match.tokens), " ".join(chunk_tokens)
      ]))



def main():
  match_map = get_match_map(sys.argv[1])
  numexamples = 100
  cursor = ordb.get_cursor("../../db/or.db")
  rows = get_pairs(numexamples, cursor)

  for row in tqdm.tqdm(list(rows)):

    review_sid = row["review_sid"]
    print(review_sid)
    rebuttal_sid = row["rebuttal_sid"]
    rule_based_matcher(review_sid, rebuttal_sid, cursor, match_map)

if __name__ == "__main__":
  main()
