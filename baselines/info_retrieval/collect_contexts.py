import corenlp
import json
import sys
import openreview

import openreview_lib as orl


def main():

  guest_client = openreview.Client(baseurl='https://api.openreview.net')
  conference = orl.Conference.iclr18
  notes = list(openreview.tools.iterget_notes(
               guest_client,
               invitation=orl.INVITATION_MAP[conference]))

  obj = {conference:[]}

  with corenlp.CoreNLPClient(
      annotators=orl.CORENLP_ANNOTATORS,
      output_format='conll') as corenlp_client:
    for note in notes:
      p = orl.get_tokenized_chunks(corenlp_client, note.content["abstract"])
      obj[conference].append(p)

  with open("tokenized_abstracts.json", 'w') as f:
    json.dump(obj, f)


if __name__ == "__main__":
  main()
