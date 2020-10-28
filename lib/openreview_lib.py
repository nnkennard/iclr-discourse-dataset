import collections
import json
import openreview

from tqdm import tqdm
import lib.openreview_db as ordb


CORENLP_ANNOTATORS = "ssplit tokenize"
Pair = collections.namedtuple("Pair", "forum review rebuttal".split())

class Conference(object):
  iclr18 = "iclr18"
  iclr19 = "iclr19"
  iclr20 = "iclr20"
  ALL = [iclr18, iclr19, iclr20]


class AuthorCategories(object):
  CONFERENCE = "Conference"
  AUTHOR = "Author"
  AC = "AreaChair"
  REVIEWER = "Reviewer"
  ANON = "Anonymous"
  NAMED = "Named"
  PC = "ProgramChair"


def shorten_author(author):
  # TODO: do this way earlier
  assert "|" not in author # Only one signature per comment, I hope
  if "Author" in author:
    return AuthorCategories.AUTHOR
  if "Conference" in author:
    return AuthorCategories.CONFERENCE
  elif "Area_Chair" in author:
    return AuthorCategories.AC
  elif "Program_Chairs" in author:
    return AuthorCategories.PC
  elif "AnonReviewer" in author:
    return AuthorCategories.REVIEWER
  elif author == "(anonymous)":
    return AuthorCategories.ANON
  else:
    assert author.startswith("~")
    return AuthorCategories.NAMED


INVITATION_MAP = {
    Conference.iclr18:'ICLR.cc/2018/Conference/-/Blind_Submission',
    Conference.iclr19:'ICLR.cc/2019/Conference/-/Blind_Submission',
    Conference.iclr20:'ICLR.cc/2020/Conference/-/Blind_Submission',
}

# This is just to have a split name available for the column
FAKE_SPLIT_MAP = {ordb.TextTables.UNSTRUCTURED: "train",
                  ordb.TextTables.TRUE_TEST: "test"}


def get_datasets(dataset_file, corenlp_client, conn, debug=False):
  """Given a dataset file, enter its reviews and rebuttals into a database. 

    Args:
      dataset_file: json file produced by make_train_test_split.py
      corenlp_client: stanford-corenlp client with at least ssplit, tokenize
      conn: connection to a sqlite3 database
      debug: set in order to truncate to 50 examples

  """
  with open(dataset_file, 'r') as f:
    dataset_obj = json.loads(f.read())

  guest_client = openreview.Client(baseurl='https://api.openreview.net')

  for subset, subset_info in dataset_obj.items():
    forum_ids = subset_info["forums"]
    conference = subset_info["conference"]

    if subset == ordb.TextTables.TRAIN_DEV:
      for set_split, set_forum_ids in forum_ids.items():
        build_dataset(set_forum_ids, guest_client, corenlp_client, conn,
                      conference, set_split, subset, debug)
    else:
      fake_split = FAKE_SPLIT_MAP[subset]
      build_dataset(forum_ids, guest_client, corenlp_client, conn,
                    conference, fake_split, subset, debug)


def flatten_signature(note):
  """Map signature field to a deterministic string.
     Tbh it looks like most signatures are actually only 1 author long..
  """
  assert len(note.signatures) == 1
  return  "|".join(sorted(sig.split("/")[-1] for sig in note.signatures))


TOKEN_INDEX = 1  # Index of token field in conll output


def get_tokens_from_tokenized(tokenized):
  """Extract token sequences from CoreNLP output.
    Args:
      tokenized: The conll-formatted output from a CoreNLP server with ssplit
      and tokenize
    Returns:
      sentences: A list of sentences in which each sentence is a list of tokens.
  """
  lines = tokenized.split("\n")
  sentences = []
  current_sentence = []
  for line in lines:
    if not line.strip(): # Blank links indicate the end of a sentence
      if current_sentence:
        sentences.append(current_sentence)
      current_sentence = []
    else:
      current_sentence.append(line.split()[TOKEN_INDEX])
  return sentences 


def get_tokenized_chunks(corenlp_client, text):
  """Runs tokenization using a CoreNLP client.
    Args:
      corenlp_client: a corenlp client with at least ssplit and tokenize
      text: raw text
    Returns:
      A list of chunks in which a chunk is a list of sentences and a sentence is
      a list of tokens.
  """
  chunks = text.split("\n")
  return [get_tokens_from_tokenized(corenlp_client.annotate(chunk))
      for chunk in chunks]


def get_forum_pairs(forum_id, note_map):
  top_children = [note
                  for note in note_map.values()
                  if note.replyto == forum_id]
  review_ids = [note.id
                for note in top_children
                if shorten_author(
                    flatten_signature(note)) == AuthorCategories.REVIEWER]
  return [Pair(forum=forum_id, review=note.replyto, rebuttal=note.id)
      for note in note_map.values()
      if (note.replyto in review_ids
          and shorten_author(
              flatten_signature(note)) == AuthorCategories.AUTHOR)]


def get_descendant_path(sid, ordered_notes):
  sid_index = None
  for i, note in enumerate(ordered_notes):
    if note.id == sid:
      sid_index = i
      root_note = note
      break
  assert sid_index is not None
  descendants = [root_note]
  for i, note in enumerate(ordered_notes):
    if i <= sid_index:
      continue
    else:
      if (note.replyto == descendants[-1].id 
              and flatten_signature(note) == flatten_signature(descendants[-1])):
        descendants.append(note)
  return [desc.id for desc in descendants[1:]]

       

def build_sid_map(note_map, forum_pairs):    
  sid_map = {}
  ordered_notes = sorted(note_map.values(), key=lambda x:x.tcdate)
  seen_notes = set()

  relevant_sids = set()
  for pair in forum_pairs:
    relevant_sids.add(pair.review)
    relevant_sids.add(pair.rebuttal)

  for i, note in enumerate(ordered_notes):
    if note.id in seen_notes:
      continue
    siblings = [sib.id
            for sib in ordered_notes[i+1:]
            if sib.replyto == note.replyto
            and flatten_signature(sib) == flatten_signature(note)]
    descendants = get_descendant_path(note.id, ordered_notes)
    if siblings and descendants: # This is too complicated to detangle
      continue
    else:
      if note.id in relevant_sids:
          notes = [note.id] + siblings + descendants
          seen_notes.update(notes)
          sid_map[note.id] = notes

  return sid_map


def get_review_rebuttal_pairs(forums, or_client):
  """From a list of forums, extract review and rebuttal pairs.
  
    Args:
      forums: A list of forum ids (directly from OR API)

    Returns:
      sid_map: A map from sids to a list of comments they encompass
      review_rebuttal_pairs: A list of pairs of sids (supernote ids)

  """
  review_rebuttal_pairs = []
  sid_map = {}
  for forum_id in forums:
    note_map = {note.id: note
            for note in or_client.get_notes(forum=forum_id)}

    forum_pairs = get_forum_pairs(forum_id, note_map)
    forum_sid_map = build_sid_map(note_map, forum_pairs)


    assert (len(sid_pairs) == len(set(sid_pairs))
            == len(set(x.review for x in sid_pairs))
            == len(set(x.rebuttal for x in sid_pairs)))

    sid_map[forum_id] = forum_sid_map
    review_rebuttal_pairs += sid_pairs
  
  return sid_map, review_rebuttal_pairs


def get_info(note):
  """Gets relevant note metadata.
    Args:
      note_id: the note id from the openreview.Note object
      note_map: a map from note ids to relevant openreview.Note objects
    Returns:
      The text_type, text and authors of the note.
  """
  if note.replyto is None:
    return "root", "", flatten_signature(note)
  else:
    for text_type in ["review", "comment", "withdrawal confirmation",
            "metareview"]:
      if text_type in note.content:
        return text_type, note.content[text_type], flatten_signature(note)
    assert False

    
def get_text_from_note_list(note_list, supernote_as_dict, corenlp_client):
  chunk_offset = 0
  text_rows = []

  for subnote in note_list:
    text_type, text, subnote_author = get_info(subnote)
    chunks = get_tokenized_chunks(corenlp_client, text)

    for chunk_idx, chunk in enumerate(chunks):
      for sentence_idx, sentence in enumerate(chunk):
        for token_idx, token in enumerate(sentence):
          new_row = ordb.TextRow(**supernote_as_dict)
          new_row.chunk_idx = chunk_idx + chunk_offset
          new_row.sentence_idx = sentence_idx
          new_row.token_idx = token_idx
          new_row.token = token
          new_row.original_id = subnote
          text_rows.append(new_row)

    chunk_offset += len(chunks)

  return text_rows

def get_parent_sid(parent_id, sid_map, forum_id):
  if parent_id in sid_map or parent_id == forum_id:
    return parent_id
  else:
    for k, v in sid_map.items():
      if parent_id in v:
        return k
  assert False

def get_comment_type(sid, reviews, rebuttals):  
  if sid in reviews:
    return "review"
  else:
    assert sid in rebuttals
    return "rebuttal"

def build_dataset(forum_ids, or_client, corenlp_client, conn, conference,
    set_split, table, debug):
  """Initializes and dumps to sqlite3 database.

  Args:
    forum_ids: list of forum ids (top comment ids) in the dataset
    or_client: openreview client
    corenlp_client: stanford-corenlp client for tokenization
    conn: sqlite3 connection
    conference: conference name
    set_split: train/dev/test
    debug: if True, truncate example list to 10 examples
  """
  all_conference_notes = openreview.tools.iterget_notes(
        or_client, invitation=INVITATION_MAP[conference])
  forums = [n.forum
            for n in all_conference_notes
            if n.forum in forum_ids]
  if debug:
    forums = forums[:10]

  sid_map, review_rebuttal_pairs = get_review_rebuttal_pairs(forums, or_client)

  reviews = [pair.review for pair in review_rebuttal_pairs]
  rebuttals = [pair.rebuttal for pair in review_rebuttal_pairs]
  
  for pair in review_rebuttal_pairs:
    forum_notes = or_client.get_notes(forum=pair.forum)
    assert forum_notes[0].id == pair.forum
    forum_title = forum_notes[0].content["title"]
    review_author, = [flatten_signature(note)
                      for note in forum_notes
                      if note.id == pair.review]
    #ordb.insert_into_pairs(conn, table, (pair.review, pair.rebuttal, set_split,
    #    forum_title, review_author) )


  # Tokenize all relevant comments
  for forum, forum_sid_map in sid_map.items():
    note_map = {note.id: note for note in or_client.get_notes(forum=forum)}
    for sid, subnotes in forum_sid_map.items():
      if sid == forum:
        dsds
      supernote = note_map[sid]
      supernote_author = flatten_signature(supernote)
      author_type = shorten_author(supernote_author)
      parent_sid = get_parent_sid(supernote.replyto, sid_map, forum)
      review_or_rebuttal = get_comment_type(sid, reviews=reviews, rebuttals=rebuttals)
      supernote_as_dict =  ordb.TextRow(
          forum, set_split, parent_sid, sid, 
          note_map[sid].tcdate, supernote_author, author_type,
          review_or_rebuttal)._asdict()
      text = get_text_from_note_list(
              [note_map[subnote] for subnote in subnotes], supernote_as_dict, corenlp_client)
      print(text)


