import collections
import re

from tqdm import tqdm

Pair = collections.namedtuple("Pair",
  "forum review_sid rebuttal_sid title review_author".split())

class Conference(object):
  iclr18 = "iclr18"
  iclr19 = "iclr19"
  iclr20 = "iclr20"
  ALL = [iclr18, iclr19, iclr20]

class Split(object):
  UNSTRUCTURED = "unstructured"
  TRAINDEV = "traindev"
  TRUETEST = "truetest"

class SubSplit(object):
  TRAIN = "train"
  DEV = "dev"
  TEST = "test"

#class DiscourseUnit(object):
#  sentence = "sentence"
#  chunk = "chunk"


DATSETS = [
    Split.UNSTRUCTURED,
    Split.TRAINDEV + "_" + SubSplit.TRAIN,
    Split.TRAINDEV + "_" + SubSplit.DEV,
    Split.TRAINDEV + "_" + SubSplit.TEST,
    Split.TRUETEST,
    ]

INVITATION_MAP = {
    Conference.iclr18:'ICLR.cc/2018/Conference/-/Blind_Submission',
    Conference.iclr19:'ICLR.cc/2019/Conference/-/Blind_Submission',
    Conference.iclr20:'ICLR.cc/2020/Conference/-/Blind_Submission',
}


class AuthorCategories(object):
  CONFERENCE = "Conference"
  AUTHOR = "Author"
  AC = "AreaChair"
  REVIEWER = "Reviewer"
  ANON = "Anonymous"
  NAMED = "Named"
  PC = "ProgramChair"



def flatten_signature(note):
  """Map signature field to a deterministic string.
     Tbh it looks like most signatures are actually only 1 author long..
  """
  assert len(note.signatures) == 1
  return  "|".join(sorted(sig.split("/")[-1] for sig in note.signatures))


TOKEN_INDEX = 1  # Index of token field in conll output



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
    relevant_sids.add(pair.review_sid)
    relevant_sids.add(pair.rebuttal_sid)

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


import datetime
def get_forum_pairs(forum_id, note_map):
  title = note_map[forum_id].content["title"]
  top_children = [note
                  for note in note_map.values()
                  if note.replyto == forum_id]
  review_ids = [note.id
                for note in top_children
                if shorten_author(
                    flatten_signature(note)) == AuthorCategories.REVIEWER]
  pairs = []
  
  for review_id in review_ids:
    candidate_responses = sorted([note
        for note in note_map.values()
        if note.replyto == review_id and shorten_author(
              flatten_signature(note)) == AuthorCategories.AUTHOR ], 
              key=lambda x:x.cdate)
    if not candidate_responses:
      continue
    else:
      super_response = candidate_responses[0]
      pairs.append(
        Pair(forum=forum_id, title=title,
             review_sid=review_id,
             rebuttal_sid=super_response.id,
             review_author=flatten_signature(
               note_map[super_response.replyto])))

  return pairs


def get_abstract_texts(forums, or_client, corenlp_client):
  """From a list of forums, extract review and rebuttal pairs.
  
    Args:
      forums: A list of forum ids (directly from OR API)

    Returns:
      sid_map: A map from sids to a list of comments they encompass

  """
  abstracts = []
  print("Getting abstracts")
  for forum_id in tqdm(forums):
    root_getter = [note
        for note in or_client.get_notes(forum=forum_id)
        if note.id == forum_id]
    assert len(root_getter) == 1
    root, = root_getter
    abstract_text = root.content["abstract"]
    abstracts.append(
        get_tokenized_chunks(corenlp_client, abstract_text))
  return abstracts




def get_review_rebuttal_pairs(forums, or_client):
  """From a list of forums, extract review and rebuttal pairs.
  
    Args:
      forums: A list of forum ids (directly from OR API)

    Returns:
      sid_map: A map from sids to a list of comments they encompass

  """
  review_rebuttal_pairs = []
  sid_map = {}
  print("Getting forums")
  for forum_id in tqdm(forums):
    note_map = {note.id: note
            for note in or_client.get_notes(forum=forum_id)}

    forum_pairs = get_forum_pairs(forum_id, note_map)
    # Sometimes these aren't supernotes e.g. when there are two candidate
    # supernotes and they have to be ordered by timestamp.
    forum_sid_map = build_sid_map(note_map, forum_pairs)
    sid_pairs = [pair 
                 for pair in forum_pairs
                 if (pair.rebuttal_sid in forum_sid_map
                     and pair.review_sid in forum_sid_map)]

    for pair in sid_pairs:
      assert pair.review_sid in forum_sid_map
      assert pair.rebuttal_sid in forum_sid_map

    assert (len(sid_pairs) == len(set(sid_pairs))
            == len(set(x.review_sid for x in sid_pairs))
            == len(set(x.rebuttal_sid for x in sid_pairs)))

    sid_map[forum_id] = forum_sid_map
    review_rebuttal_pairs += sid_pairs

  full_sid_map = {}
  for forum_id, forum_sid_map in sid_map.items():
    id_to_note_map = {note.id:note
        for note in or_client.get_notes(forum=forum_id)} 
    full_sid_map[forum_id] = {k:[id_to_note_map[i] for i in v]
                              for k, v in forum_sid_map.items()}

  return full_sid_map, review_rebuttal_pairs

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

#NEWLINE_PLACEHOLDER = "NEWLINE_PLACEHOLDER"
#NEWLINE_PLACEHOLDER_PADDED = " " +  NEWLINE_PLACEHOLDER + " "

def get_tokenized_chunks(corenlp_client, text):
  """Runs tokenization using a CoreNLP client.
    Args:
      corenlp_client: a corenlp client with at least ssplit and tokenize
      text: raw text
    Returns:
      A list of chunks in which a chunk is a list of sentences and a sentence is
      a list of tokens.
  """
  chunk_texts = [chunk.strip() for chunk in text.split("\n")]
  return [get_tokens_from_tokenized(corenlp_client.annotate(chunk_text))
          for chunk_text in chunk_texts]


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


    
def get_text_from_note_list(note_list, corenlp_client):
  """Given a list of notes, get their text, and tokenize it.

    Args:
      note_list: A list of note ids
      corenlp_client: A corenlp client with at least ssplit and tokenize

    Returns:
      A list of chunks. Each chunk is a list of sentences. Each sentence is a
      list of tokens.

  """
  supernote_text = []

  for subnote in note_list:
    text_type, text, subnote_author = get_info(subnote)
    supernote_text += get_tokenized_chunks(corenlp_client, text)

  for chunk in supernote_text:
    print("Chunk")
    for sentence in chunk:
      print("Sentence")
      print(" ".join(sentence))

  return supernote_text

Example = collections.namedtuple("Example",
  ("index review_sid rebuttal_sid review_text rebuttal_text "
   "title review_author forum labels").split())


def get_pair_text(pairs, sid_map, corenlp_client):
  examples = []

  print("Processing pairs")
  for i, pair in tqdm(list(enumerate(pairs))):
    review_text = get_text_from_note_list(
        sid_map[pair.forum][pair.review_sid], corenlp_client)
    rebuttal_text = get_text_from_note_list(
        sid_map[pair.forum][pair.rebuttal_sid], corenlp_client)
    examples.append(Example(
      i, pair.review_sid, pair.rebuttal_sid, review_text, rebuttal_text,
      pair.title, pair.review_author, pair.forum, None)._asdict())

  return examples
