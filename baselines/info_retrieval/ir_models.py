import argparse
import collections
import json
import torch

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Load OpenReview data from a sqlite3 database.')
parser.add_argument('-d', '--dataset_file', type=str,
                    help='path to dataset file')


class IRModelInterface(object):
  def __init__(self):
    pass

  def run(self, dataset):
    pass



def segments_to_tokens(segments, segment_type):
  if segment_type == "chunk":
    return [sum(segment, []) for segment in segments]
  else:
    assert segment_type == "sentence"
    return segments


Result = collections.namedtuple("Result", "review_sid rebuttal_sid labels")

class BertVSM(IRModelInterface):
  def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
    self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')

  def run(self, dataset):
    results = []
    segment_type = dataset["examples"][0]["seg_format"]
    print("Encoding examples")
    for example in tqdm(dataset["examples"]):
      
      review_vectors = self._encode_segments(example["review_text"],
          segment_type)
      rebuttal_vectors = self._encode_segments(example["rebuttal_text"],
                                         segment_type)
      similarities = review_vectors.matmul(rebuttal_vectors.transpose(0,1))
      example_labels = []
      for rebuttal_index in range(rebuttal_vectors.shape[0]):
        top_k = torch.topk(similarities[:,rebuttal_index],
          k=min(3,len(example["review_text"]))).indices.data.tolist()
        example_labels.append(top_k)
      results.append(Result(example["review_sid"], example["rebuttal_sid"],
        example_labels)._asdict())

    return results


  def _encode_segments(self, segments, segment_type):
    tokens = segments_to_tokens(segments, segment_type)
    vectors = []
    for token_segment in tokens:
      input_tensors = torch.tensor(
          self.tokenizer.encode(token_segment)).unsqueeze(0)
      vectors.append(self.model(input_tensors)[0].mean(1))
    return torch.cat(vectors)


def main():
  args = parser.parse_args()

  with open(args.dataset_file, 'r') as f:
    dataset = json.load(f)
  bert_model = BertVSM()
  results = bert_model.run(dataset)

  assert args.dataset_file.endswith(".json")
  output_file = args.dataset_file.replace(".json", "_bert-results.json")
  with open(output_file, 'w') as f:
    json.dump(results, f)

  print(results)


if __name__ == "__main__":
  main()
