import collections
import sqlite3
from sqlite3 import Error

from recordtype import recordtype


def create_connection(db_file):
  """ create a database connection to a SQLite database """
  conn = None
  try:
    conn = sqlite3.connect(db_file)
    conn.row_factory = dict_factory
    return conn
  except Error as e:
    print(e)

def dict_factory(cursor, row):
  d = {}
  for idx, col in enumerate(cursor.description):
    d[col[0]] = row[idx]
  return d

FIELDS = """forum_id split parent_supernote comment_supernote
  timestamp author author_type or_text_type
  comment_type original_id chunk_idx sentence_idx
  token_idx token"""
COMMA_SEP_FIELDS = ", ".join(FIELDS.split())
CommentRow = recordtype("CommentRow", FIELDS, default=None)


class TextTables(object):
  UNSTRUCTURED = "unstructured"
  TRAIN_DEV = "traindev"
  TRUE_TEST = "truetest"
  ALL = [UNSTRUCTURED, TRAIN_DEV, TRUE_TEST]


def create_tables(conn):
  """ create a table from the create_table_sql statement
  :param conn: Connection object
  :param create_table_sql: a CREATE TABLE statement
  :return:
  """
  try:
    c = conn.cursor()
    for table_name in TextTables.ALL:
      c.execute(CREATE_TEXT_TABLE.format(table_name))
      c.execute(CREATE_PAIR_TABLE.format(table_name + "_pairs"))
    conn.commit()
  except Error as e:
    print(e)


CREATE_PAIR_TABLE =  """ CREATE TABLE IF NOT EXISTS {0} (
    review_supernote text NOT NULL,
    rebuttal_supernote text NOT NULL,
    split text NOT NULL,

    PRIMARY KEY (review_supernote, rebuttal_supernote)); """

CREATE_TEXT_TABLE = """ CREATE TABLE IF NOT EXISTS {0} (
    forum_id text NOT NULL,
    split text NOT NULL,
    parent_supernote text NOT NULL,
    comment_supernote text NOT NULL,

    timestamp text NOT NULL,
    author text NOT NULL,
    author_type text NOT NULL,
    or_text_type text NOT NULL,
    comment_type text NOT NULL,

    original_id text NOT NULL,
    chunk_idx integer NOT NULL,
    sentence_idx integer NOT NULL,
    token_idx integer NOT NULL,
    token text NOT NULL,

    PRIMARY KEY (original_id, chunk_idx, sentence_idx, token_idx)); """


def insert_into_pairs(conn, table, pair_rows):
  """Insert a record into the datasets table (train-test split)."""
  cmd = ''' INSERT INTO {0} (review_supernote, rebuttal_supernote, split)
              VALUES(?, ?, ?); '''.format(table)
  cur = conn.cursor()
  for row in pair_rows:
    cur.execute(cmd, row)
  conn.commit()



def insert_into_comments(conn, table, forum_rows):
  """Insert a record into the datasets table (train-test split)."""
  cmd = ''' INSERT INTO
              {0}({1})
              VALUES(?, ?, ?, ?,
                     ?, ?, ?, ?,
                     ?, ?, ?, ?,
                     ?, ?); '''.format(table, COMMA_SEP_FIELDS)
  cur = conn.cursor()
  for row in forum_rows:
    cur.execute(cmd, tuple(row))
  conn.commit()


def collapse_dict(input_dict):
  assert sum(input_dict.keys()) + 5 > 0
  # This is a garbage way to get an error if the keys are not ints
  return [input_dict[i] for i in sorted(input_dict.keys())]


def crunch_text_rows(rows):
  """Crunch rows from text table back into a more readable format.

  TODO(nnk): This, but in a non-horrible way
  """
  texts_builder = collections.defaultdict(lambda : collections.defaultdict(lambda:
    collections.defaultdict(list)))

  for row in rows:
    supernote, chunk, sentence, token = (row["comment_supernote"],
        row["chunk"], row["sentence"], row["token"])
    texts_builder[supernote][chunk][sentence].append(token)

  texts = {}
  for supernote, chunk_dict in texts_builder.items():
    texts[supernote] = [collapse_dict(sentence) for sentence in 
    collapse_dict(chunk_dict)]

  return texts
