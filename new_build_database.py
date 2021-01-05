import collections
import openreview
import sys

import new_openreview_lib as norl

ForumList = collections.namedtuple("ForumList",
                                   "conference forums url".split())

def get_unstructured_ids(conference, client):
  #return get_sampled_forums(conference, client, 1)
  return get_sampled_forums(conference, client, 0.01)


def get_sampled_forums(conference, client, sample_rate):
  forums = [forum.id
            for forum in get_all_conference_forums(conference, client)]
  if sample_rate == 1:
    pass
  else:
    random.shuffle(forums)
    forums = forums[:int(sample_rate * len(forums))]
  return ForumList(conference, forums, norl.INVITATION_MAP[conference])

def get_all_conference_forums(conference, client):
  return list(openreview.tools.iterget_notes(
    client, invitation=norl.INVITATION_MAP[conference]))



def main():
  guest_client = openreview.Client(baseurl='https://api.openreview.net')

  # Declare unstructured/traindev/truetest

  SPLIT_TO_CONFERENCE = {
    norl.Split.UNSTRUCTURED: norl.Conference.iclr18,
    norl.Split.TRAINDEV: norl.Conference.iclr19,
    norl.Split.TRUETEST: norl.Conference.iclr20
    }

  # For unstructured:
  unstruct_conference = SPLIT_TO_CONFERENCE[norl.Split.UNSTRUCTURED]
  unstruct_forums = get_unstructured_ids(unstruct_conference, guest_client)
  unstruct_pair_text = norl.get_review_rebuttal_pairs(
      unstruct_forums.forums, guest_client)

    # Get valid review-rebuttal pairs

    # Get abstracts
  # For traindev
    # Get train dev split (no stratification)
    # For each split
      # Get valid review-rebuttal pairs
  # For truetest
    # Get valid review-rebuttal pairs


if __name__ == "__main__":
  main()
