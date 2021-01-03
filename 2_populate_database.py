import argparse
import glob
import os
import sys

import openreview_db as ordb
import openreview_lib as orl


parser = argparse.ArgumentParser(
    description='Load OpenReview data into a sqlite3 database.')
parser.add_argument('-d', '--dbfile', default="db/or.db",
    type=str, help='path to database file')
parser.add_argument('-i', '--inputfile',
    default="splits/iclr_discourse_dataset_split.json",
    type=str, help='path to database file')
parser.add_argument('-s', '--debug', action="store_true",
    help='if True, truncate the example list')
parser.add_argument('-c', '--clean', action="store_true",
    help='if True, delete existing database and intermediate data structures')


def main():

  args = parser.parse_args()
  conn = ordb.create_connection(args.dbfile)
  if args.clean:
    for filename in glob.glob("db/*"):# + glob.glob("/iesl/canvas/nnayak/temp/or_ir/*"):
      try:
        print("Deleting ", filename)
        os.remove(filename)
      except:
        print("Alas, an error")
  if conn is not None:
    ordb.create_tables(conn)
    orl.get_datasets(args.inputfile, conn, debug=args.debug)  
  else:
    print("Database connection error.")


if __name__ == "__main__":
  main()
