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


class TextTables(object):
  UNSTRUCTURED = "unstructured"
  TRAIN_DEV = "traindev"
  TRUE_TEST = "truetest"
  ALL = [UNSTRUCTURED, TRAIN_DEV, TRUE_TEST]


PAIR_FIELDS = "review_sid rebuttal_sid split title review_author".split()
PairRow = recordtype("TextRow", PAIR_FIELDS, default=None)

TEXT_FIELDS = ("forum_id split parent_sid sid timestamp author "
                     "author_type comment_type orig_id chunk_idx sentence_idx "
                     "token_idx token").split() # These are all "text NOT NULL"
TextRow = recordtype("TextRow", TEXT_FIELDS, default=None)


def create_tables(conn):
  """ create a table from the create_table_sql statement
  :param conn: Connection object
  :param create_table_sql: a CREATE TABLE statement
  :return:
  """
  try:
    c = conn.cursor()
    for table_name in TextTables.ALL:

      create_text_table_cmd =  "CREATE TABLE IF NOT EXISTS {0} ("
      for field in TEXT_FIELDS:
        create_text_table_cmd +=  field + " text NOT NULL, "
      create_text_table_cmd += ("PRIMARY KEY (orig_id, "
                      "chunk_idx, sentence_idx, token_idx)); ")
      c.execute(create_text_table_cmd.format(table_name))

      create_pair_table_cmd =  "CREATE TABLE IF NOT EXISTS {0} ("
      for field in PAIR_FIELDS:
        create_pair_table_cmd +=  field + " text NOT NULL, "
      create_pair_table_cmd += ("PRIMARY KEY (review_sid, rebuttal_sid)); ")
      c.execute(create_pair_table_cmd.format(table_name + "_pairs"))

    conn.commit()
  except Error as e:
    print(e)


def insert_into_table(conn, table_name, fields, rows):
  cmd = "".join(["INSERT INTO {0}({1}) VALUES (",
                 ",".join(["?"] * len(fields)),
                ");"]).format(table_name, ",".join(fields))
  print(table_name)
  print(fields)
  print(rows[0])
  cur = conn.cursor()
  for row in rows:
    cur.execute(cmd, tuple(row))
  conn.commit()


def collapse_dict(input_dict):
  return [input_dict[i] for i in sorted(input_dict.keys())]


def crunch_text_rows(rows):
  """Crunch rows from text table back into a more readable format.

  TODO(nnk): This, but in a non-horrible way
  """
  texts_builder = collections.defaultdict(
          lambda : collections.defaultdict(lambda:
                   collections.defaultdict(list)))

  for row in rows:
    supernote, chunk, sentence, token = (row["sid"],
        row["chunk_idx"], row["sentence_idx"], row["token"])
    texts_builder[supernote][chunk][sentence].append(token)

  texts = {}
  for supernote, chunk_dict in texts_builder.items():
    texts[supernote] = [collapse_dict(sentence) 
        for sentence in collapse_dict(chunk_dict)]

  return texts
