import argparse
import corenlp
import sys

import lib.openreview_db as ordb
import lib.openreview_lib as orl


parser = argparse.ArgumentParser(
    description='Load OpenReview data into a sqlite3 database.')
parser.add_argument('-d', '--dbfile', default="db/or.db",
    type=str, help='path to database file')
parser.add_argument('-i', '--inputfile',
    default="splits/iclr_discourse_dataset_split.json",
    type=str, help='path to database file')
parser.add_argument('-s', '--debug', action="store_true",
    help='if True, truncate the example list')


def main():

  args = parser.parse_args()
  conn = ordb.create_connection(args.dbfile)
  if conn is not None:
    ordb.create_tables(conn)
    with corenlp.CoreNLPClient(
        annotators=orl.ANNOTATORS, output_format='conll') as corenlp_client:
      orl.get_datasets(
          args.inputfile, corenlp_client, conn, debug=args.debug)  
  else:
    print("Database connection error.")


if __name__ == "__main__":
  main()
