import collections
from difflib import SequenceMatcher
import openreview
import random
import re

from tqdm import tqdm


from spacy.lang.en import English
spacy_nlp = English()
spacy_nlp.add_pipe("sentencizer")


random.seed(47)

ForumList = collections.namedtuple("ForumList",
                                   "conference forums url".split())

Pair = collections.namedtuple(
    "Pair", "forum review_sid rebuttal_sid title review_author".split())

Example = collections.namedtuple(
    "Example", ("index review_sid rebuttal_sid review_text rebuttal_text "
                "title review_author forum labels exact_matches").split())

Sentence = collections.namedtuple("Sentence", "start_index end_index suffix")
Comment = collections.namedtuple("Comment", "text sentences")

def make_sentence_dict(start, end, suffix):
  return Sentence(start, end, suffix)._asdict()


ClassificationExample = collections.namedtuple(
    "ClassificationExample",
    ("index sid text forum_title top_comment_title  author forum labels"
     ).split())


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


DATASETS = [
    Split.UNSTRUCTURED,
    Split.TRAINDEV + "_" + SubSplit.TRAIN,
    Split.TRAINDEV + "_" + SubSplit.DEV,
    Split.TRAINDEV + "_" + SubSplit.TEST,
    Split.TRUETEST,
]

INVITATION_MAP = {
    Conference.iclr18: 'ICLR.cc/2018/Conference/-/Blind_Submission',
    Conference.iclr19: 'ICLR.cc/2019/Conference/-/Blind_Submission',
    Conference.iclr20: 'ICLR.cc/2020/Conference/-/Blind_Submission',
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
  return "|".join(sorted(sig.split("/")[-1] for sig in note.signatures))


def shorten_author(author):
  # TODO: do this way earlier
  assert "|" not in author  # Only one signature per comment, I hope
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
  """Get a path of descendants that share the same author.

     Args:
       sid: super id of the ancestor in question
       ordered_notes: notes in the forum ordered by creation date

     Returns:
       A list of the note ids in the path of descendants
  """

  # Find the super note in the time ordering
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
      # we expect descendants to be created after...
      # not always true unfortunately
      continue
    else:
      if (note.replyto == descendants[-1].id
          and flatten_signature(note) == flatten_signature(descendants[-1])):
        descendants.append(note)
  return [desc.id for desc in descendants[1:]]


def build_sid_map(note_map, forum_pairs):
  """Builds a map from top comment ids to their continuation descendants.
    
     Args:
       note_map: map from note.ids to openreview.Note objects for one forum
       forum_pairs: possible (review, rebuttal) pairs in the forum
  """
  sid_map = {}
  ordered_notes = sorted(note_map.values(), key=lambda x: x.tcdate)
  # Order all notes (comments) in a forum by their time of creation.
  # Note: sometimes this results in clearly out-of-order comments.
  # This needs to be addressed manually on a case-by-case basis.

  relevant_sids = set()
  # Gather definite super_ids (top comment structurally of a review or rebuttal)
  for pair in forum_pairs:
    relevant_sids.add(pair.review_sid)
    relevant_sids.add(pair.rebuttal_sid)

  seen_notes = set()
  for i, note in enumerate(ordered_notes):
    if note.id in seen_notes:
      continue
    siblings = [
        sib.id for sib in ordered_notes[i + 1:] if sib.replyto == note.replyto
        and flatten_signature(sib) == flatten_signature(note)
    ]
    # Siblings by the same author
    descendants = get_descendant_path(note.id, ordered_notes)
    # Descendants by the same author (no breaks)

    if siblings and descendants:  # This is too complicated to detangle
      continue
    else:
      # Otherwise, add siblings or descendants of the super note to the map.
      if note.id in relevant_sids:
        notes = [note.id] + siblings + descendants
        seen_notes.update(notes)
        sid_map[note.id] = notes

  return sid_map


def get_forum_pairs(forum_id, note_map):
  """ Gets (review, rebuttal) pairs from a forum.

      Args:
        forum_id: The forum id from OpenReview
        note_map: A map from note_id to openreview.Note objects for all the
        notes in the forum.

      Returns:
        A list of Pairs containing a review and a rebuttal (by super id)
  """

  title = note_map[forum_id].content["title"]
  top_children = [
      note for note in note_map.values() if note.replyto == forum_id
  ]
  review_ids = [
      note.id for note in top_children
      if shorten_author(flatten_signature(note)) == AuthorCategories.REVIEWER
      and 'review' in note.content
  ]
  # Replies to the dummy 'Forum' node that are written by a Reviewer and have a
  # "review" item in their content (so not just a comment by a reviewer) count
  # as reviews.
  pairs = []

  for review_id in review_ids:
    candidate_responses = sorted([
        note for note in note_map.values() if note.replyto == review_id
        and shorten_author(flatten_signature(note)) == AuthorCategories.AUTHOR
    ],
                                 key=lambda x: x.cdate)
    # Candidate responses are responses to a known review note which are written
    # by the Authors.
    if not candidate_responses:
      continue
    else:
      super_response = candidate_responses[0]  # This should be the earliest
      pairs.append(
          Pair(forum=forum_id,
               title=title,
               review_sid=review_id,
               rebuttal_sid=super_response.id,
               review_author=flatten_signature(
                   note_map[super_response.replyto])))

  return pairs


def get_review_rebuttal_pairs(forums, or_client):
  """From a list of forums, extract review and rebuttal pairs.
  
    Args:
      forums: A list of forum ids (can be directly from OR API)

    Returns:
      sid_map: A map from "super-ids" to a list of comments they encompass

  """

  review_rebuttal_pairs = []
  sid_map = {}
  # A sid or "super-id" is used to group utterances that spread over multiple
  # OpenReview comments. For example, a reviewer replies to themselves with
  # further details, or an author adds a very long rebuttal in multiple replies
  # to the review comment.

  print("Getting forums")
  for forum_id in tqdm(forums):
    note_map = {note.id: note for note in or_client.get_notes(forum=forum_id)}

    forum_pairs = get_forum_pairs(forum_id, note_map)
    # Sometimes these aren't supernotes e.g. when there are two candidate
    # supernotes and they have to be ordered by timestamp.
    forum_sid_map = build_sid_map(note_map, forum_pairs)
    sid_pairs = [
        pair for pair in forum_pairs if (pair.rebuttal_sid in forum_sid_map
                                         and pair.review_sid in forum_sid_map)
    ]
    # Above: ensuring that we don't have pairs for which the note has actually
    # been deleted.

    for pair in sid_pairs:
      assert pair.review_sid in forum_sid_map
      assert pair.rebuttal_sid in forum_sid_map  # Double checking, lol

    assert (len(sid_pairs) == len(set(sid_pairs))  # Triple checking
            == len(set(x.review_sid for x in sid_pairs)) == len(
                set(x.rebuttal_sid for x in sid_pairs)))

    sid_map[forum_id] = forum_sid_map
    review_rebuttal_pairs += sid_pairs

  full_sid_map = {}  # This is a map for sids from all the forums
  for forum_id, forum_sid_map in sid_map.items():
    id_to_note_map = {
        note.id: note
        for note in or_client.get_notes(forum=forum_id)
    }
    full_sid_map[forum_id] = {
        k: [id_to_note_map[i] for i in v]
        for k, v in forum_sid_map.items()
    }

  return full_sid_map, review_rebuttal_pairs


def get_text(note):
  """Gets relevant note metadata.
    Args:
      note_id: the note id from the openreview.Note object
      note_map: a map from note ids to relevant openreview.Note objects
    Returns:
      The text_type, text and authors of the note.
  """
  if note.replyto is None:
    assert False
    return ""
  else:
    for text_type in [
        "review", "comment", "withdrawal confirmation", "metareview"
    ]:
      if text_type in note.content:
        return note.content[text_type]
    assert False


def get_text_from_note_list(note_list):
  supernote_text = "\n\n".join(get_text(subnote) for subnote in note_list)
  return Text(supernote_text).text


class Text(object):
  def __init__(self, text):
    sentence_texts = []
    sentence_indices = []
    for chunk in text.split("\n"):
      doc = spacy_nlp(chunk)
      for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
          continue
        if sentence_indices:
          offset = sentence_indices[-1][1]
          index = offset + text[offset:].find(sentence_text)
        else:
          index = text.find(sentence_text)
        sentence_texts.append(sentence_text)
        sentence_indices.append((index, index + len(sentence_text)))
    assert len(sentence_texts) == len(sentence_indices)
    final_sentences = []
    for i in range(len(sentence_texts) - 1):
      start, end = sentence_indices[i]
      sentence_text = sentence_texts[i]
      next_sentence_start = sentence_indices[i+1][0]
      suffix = "\n" * text[end:next_sentence_start].count("\n")
      assert sentence_text == text[start:end]
      final_sentences.append(make_sentence_dict(start, end, suffix))

    if sentence_indices:
        final_start, final_end = sentence_indices[-1]
        final_sentences.append(make_sentence_dict(final_start, final_end, ""))
    
    self.text = Comment(text, final_sentences)._asdict()



def match_wrapper(sent1, sent2):
  return SequenceMatcher(None, sent1, sent2).find_longest_match(
    0, len(sent1), 0, len(sent2))


class CommentNavigator(object):
  def __init__(self, json_rep):
    self.text = json_rep["text"]
    self.sentences = [Sentence(
    sent["start_index"], sent["end_index"], sent["suffix"],
        ) for sent in json_rep["sentences"]]
            
  def sentence_texts(self):
    return [self.text[sent.start_index:sent.end_index]+sent.suffix for sent in self.sentences]


MatchInfo = collections.namedtuple("MatchInfo",
  "review_sentence rebuttal_sentence review_offset rebuttal_offset size".split())

    
def get_exact_matches(review_text, rebuttal_text):
  review = CommentNavigator(review_text)
  rebuttal = CommentNavigator(rebuttal_text)
  review_sentences = review.sentence_texts()
  rebuttal_matches = []
  for reb_i, reb_sent in enumerate(rebuttal.sentence_texts()):
    best_match = None
    for rev_i, rev_sent in enumerate(review_sentences):
      match = match_wrapper(rev_sent, reb_sent)
      if match.size < 5:
        continue
      if best_match is None or best_match["size"] < match.size :
        best_match = MatchInfo(rev_i, reb_i, match.a, match.b,
                match.size)._asdict()
    if best_match is not None:
        rebuttal_matches.append(best_match)
  return rebuttal_matches
 
    
def get_classification_labels(notes):
  top_comment = notes[0]
  labels = {}
  for key in ["rating", "confidence"]:
    if key in top_comment.content:
      labels[key] = int(top_comment.content[key].split(":")[0])
  return labels


def get_pair_text(pairs, sid_map):
  """ Get review and rebuttal text along with metadata and labels.
      
      Args:
        pairs: A list of review/rebuttal Pairs
        sid_map: A map from super ids to the list of comments they encompass
        pipeline: A Stanza pipeline with at least ssplit, tokenize

      Returns:
        A list of Examples
  """

  examples = []

  for i, pair in tqdm(list(enumerate(pairs))):
    review_text = get_text_from_note_list(sid_map[pair.forum][pair.review_sid])
    rebuttal_text = get_text_from_note_list(
        sid_map[pair.forum][pair.rebuttal_sid])
    exact_matches = get_exact_matches(review_text, rebuttal_text)
    examples.append(
        Example(
            i, pair.review_sid, pair.rebuttal_sid, review_text, rebuttal_text,
            pair.title, pair.review_author, pair.forum,
            get_classification_labels(
                sid_map[pair.forum][pair.review_sid]),
            exact_matches)._asdict())

  return examples


def get_abstract_texts(forums, or_client):
  """From a list of forums, extract review and rebuttal pairs.
  
    Args:
      forums: A list of forum ids (directly from OR API)
      or_client: OpenReview API client
      pipeline: A Stanza pipeline with at least ssplit, tokenize

    Returns:
      sid_map: A map from sids to a list of comments they encompass

  """
  abstracts = []
  print("Getting abstracts")
  for forum_id in tqdm(forums):
    root_getter = [
        note for note in or_client.get_notes(forum=forum_id)
        if note.id == forum_id
    ]
    assert len(root_getter) == 1
    root, = root_getter
    abstract_text = root.content["abstract"]
    abstracts.append(Text(abstract_text).text)
  return abstracts


def get_all_conference_forums(conference, client):
  return list(
      openreview.tools.iterget_notes(client,
                                     invitation=INVITATION_MAP[conference]))


def get_sampled_forums(conference, client, sample_rate):
  """ Return forums from a conference, possibly sampled.

      Args:
        conference: Conference name (from openreview_lib.Conference)
        guest_client: OpenReview API guest client
        sample_rate: Fraction of forums to retain
      Returns:
        ForumList containing sampled forums and metadata
  """
  forums = [
      forum.id for forum in get_all_conference_forums(conference, client)
  ]
  if sample_rate == 1:
    pass  # Just send everything through
  else:
    random.shuffle(forums)
    forums = forums[:int(sample_rate * len(forums))]
  return ForumList(conference, forums, INVITATION_MAP[conference])


def get_sub_split_forum_map(conference, guest_client, sample_frac):
  """ Randomly sample forums into train/dev/test sets.
      
      Args:
        conference: Conference name (from openreview_lib.Conference)
        guest_client: OpenReview API guest client
        sample_frac: Percentage of examples to retain

      Returns:
        sub_split_forum_map: Map from  "train"/"dev"/"test" to a list of forum
        ids
  """

  forums = get_sampled_forums(conference, guest_client, sample_frac).forums
  random.shuffle(forums)
  offsets = {
      SubSplit.DEV: (0, int(0.2 * len(forums))),
      SubSplit.TRAIN: (int(0.2 * len(forums)), int(0.8 * len(forums))),
      SubSplit.TEST: (int(0.8 * len(forums)), int(1.1 * len(forums)))
  }
  sub_split_forum_map = {
      sub_split: forums[start:end]
      for sub_split, (start, end) in offsets.items()
  }
  return sub_split_forum_map