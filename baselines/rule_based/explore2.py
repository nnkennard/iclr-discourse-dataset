import collections
import json
import sys
import tqdm
import re

from explore import Location, Match

import trie
import openreview_db as ordb

START = """
<!-- CSS -->
<link rel="stylesheet"
href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css"
integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2"
crossorigin="anonymous">
<!-- jQuery and JS bundle w/ Popper.js -->
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"
integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj"
crossorigin="anonymous"></script>
<script
src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/js/bootstrap.bundle.min.js"
integrity="sha384-ho+j7jyWK8fNQe+A12Hb8AhRq26LrZ/JpcUGGOn+Y7RsweNrtN/tE3MoK7ZeZDyx"
crossorigin="anonymous"></script>
<link rel="stylesheet" href="{{ site.baseurl}}/assets/css/index.css">
<body class="blue">
   <br />
      <div class="container">
"""

END = """
</div>
</body>
"""

TABLE_START = """
<table class="table">
 <thead>
    <tr>
       <td> Review </td>
       <td> Rebuttal </td>
    </tr>
 </thead>
 <tbody>
"""

TABLE_END = """
   </tbody>
</table>
"""

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
    

def main():
  match_map = get_match_map(sys.argv[1])
  numexamples = 1000
  if numexamples == -1:
      get_rows_command = "SELECT * FROM traindev_pairs WHERE split=?"
  else:
      get_rows_command = ("SELECT * FROM traindev_pairs WHERE "
      "split=? LIMIT {0}").format(numexamples)
  conn = ordb.create_connection("../../db/or.db")

  if conn is None:
    print("Error")
    return
  cur = conn.cursor()
  cur.execute(get_rows_command, ("train",))
  rows = cur.fetchall()

  page_str = "" + START
  
  for row in tqdm.tqdm(list(rows)):

    review_sid = row["review_sid"]
    print(review_sid)
    rebuttal_sid = row["rebuttal_sid"]
    cur.execute("SELECT * FROM traindev WHERE sid=?",
        (review_sid,))
    review_chunks, = ordb.crunch_text_rows(cur.fetchall()).values()
    for chunk in review_chunks:
          cur.execute("SELECT * FROM traindev WHERE sid=?",
        (rebuttal_sid,))
    rebuttal_chunks, = ordb.crunch_text_rows(cur.fetchall()).values()


    find_prefixes(rebuttal_chunks)

    matches = match_map[review_sid]
    

    for match in matches:
      if len(match.tokens) < 8:
        continue
      page_str += TABLE_START
      highlighted_review = highlight(
          review_chunks, match.review_location, len(match.tokens))
      highlighted_rebuttal = highlight(
          rebuttal_chunks, match.rebuttal_location, len(match.tokens))
      page_str += " ".join(["<tr><td>", highlighted_review, "</td><td>",
                            highlighted_rebuttal, "</tr></td> <br />"])
      page_str += TABLE_END

  page_str += END
  #print(page_str)
 
if __name__ == "__main__":
  main()
