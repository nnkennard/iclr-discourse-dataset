import argparse
import sqlite3
import sys

import lib.openreview_db as ordb

parser = argparse.ArgumentParser(
    description='Load OpenReview data from a sqlite3 database.')
parser.add_argument('-d', '--dbfile', default="db/or.db",
    type=str, help='path to database file')
parser.add_argument('-n', '--numexamples', default=-1,
    type=int, help='number of examples to dump as string. If -1, dump all')


def main():

  args = parser.parse_args()
  conn = ordb.create_connection(args.dbfile)
  if conn is not None:
    cur = conn.cursor()


    if args.numexamples == -1:
        get_rows_command = "SELECT * FROM traindev_pairs WHERE split=?"
    else:
        get_rows_command = ("SELECT * FROM traindev_pairs WHERE "
        "split=? LIMIT {0}").format(args.numexamples)

    cur.execute(get_rows_command, ("train",))
    rows = cur.fetchall()
    for row in rows:
      cur.execute("SELECT * FROM traindev WHERE comment_supernote=?",
          (row["review_supernote"],))
      crunched_rows = ordb.crunch_text_rows(cur.fetchall())

      for note_id, chunks in crunched_rows.items():
        print(note_id)
        print("-" * 80)
        for chunk in chunks:
          for sentence in chunk:
            print(" ".join(sentence))
          print()
        print("*" * 80)


if __name__ == "__main__":
  main()
