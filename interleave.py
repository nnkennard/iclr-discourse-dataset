import argparse
import json
import math
import openreview_lib as orl 

parser = argparse.ArgumentParser(
    description='Build database of review-rebuttal pairs')
parser.add_argument('-i', '--input_dir', default="data/review_rebuttal_pair_dataset/",
    type=str, help='path to database file')


def main():

    args = parser.parse_args()
    id_map = {}
    
    for dataset in orl.DATASETS:
        if dataset == orl.Split.UNSTRUCTURED:
            continue
        else:
            with open("".join(
                [args.input_dir, "/", dataset, ".json"]), 'r') as f:
                obj = json.load(f)
            forum_list = []
            for pair in obj["review_rebuttal_pairs"]:
                forum_id = pair["forum"]
                if forum_id not in forum_list:
                    forum_list.append(forum_id)

            id_map[dataset] = forum_list

    min_len = min(len(i) for i in id_map.values())
    len_ratio_map = {}
    for dataset, ids in id_map.items():
        len_ratio_map[dataset] = math.ceil(len(ids)/min_len)

    interleaved = []

    for i in range(min_len):
        for dataset, ids in id_map.items():
            interleaved += ids[i * len_ratio_map[dataset]: (i+1) * len_ratio_map[dataset]]

    assert len(interleaved) == sum(len(i) for i in id_map.values())

    with open(args.input_dir + "/interleaved.json", 'w') as f:
        json.dump({"interleaved_forum_ids":interleaved}, f)


if __name__ == "__main__":
    main()
