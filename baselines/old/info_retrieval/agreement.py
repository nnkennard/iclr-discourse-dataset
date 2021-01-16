import argparse
import collections
import glob
import json

from scipy import stats

parser = argparse.ArgumentParser(
    description='Load OpenReview data from a sqlite3 database.')
parser.add_argument('-d', '--dataset_file', type=str,
                    help='path to dataset file')


ResultMap = collections.namedtuple("ResultMap", "model results".split())

def get_results_from_file(filename):
  assert filename.endswith(".json")
  input_files = [i for i in glob.glob(filename[:-5]+"*")
                 if "result" in i]
  print(filename)
  print(input_files)
  overall_results = {}
  for input_file in input_files:
    model_name = input_file[:-5].split("_")[-1]
    with open(input_file, 'r') as f:
      model_results = {}
      results = json.load(f)
      for result in results:
        model_results[
            result["review_sid"] + "_" + result["rebuttal_sid"]] = result["labels"]
      overall_results[model_name] = model_results

  print(overall_results.keys())

  correlations = {}
  for example_id, bm25_results in overall_results["bm25-results"].items():
    bert_results = overall_results["bert-results"][example_id]
    for bert_result, bm25_result in zip(bert_results, bm25_results):
      correlations[example_id] = stats.spearmanr(bert_result, bm25_result).correlation
  return correlations




def main():

  args = parser.parse_args()

  assert "chunk" in args.dataset_file
  chunk_correlations = get_results_from_file(args.dataset_file)
  sentence_correlations = get_results_from_file(args.dataset_file.replace(
                                                "chunk", "sentence"))
  for example_id, chunk_correlation in chunk_correlations.items():
    print(example_id, chunk_correlation, sentence_correlations[example_id])


if __name__ == "__main__":
  main()
