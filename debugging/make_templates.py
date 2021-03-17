import csv
import json
import os
import sys

def get_text_as_string(tokenized_text, review_or_rebuttal):
    if review_or_rebuttal == "review":
        prefix = "V"
    else:
        assert review_or_rebuttal == "rebuttal"
        prefix = "B"
    text_builder = []
    for para in tokenized_text:
        for sentence in para:
            text_builder.append(" ".join(sentence).encode("utf-8"))
        text_builder.append("")

    return [
            (prefix + str(i), sentence)
            for i, sentence in enumerate(text_builder)]

def pad(text, max_len):
    return text + [("", "")] * (max_len - len(text))

FIELD_NAMES = ("Review_index Review_sentence Review_affordance - "
               "Rebuttal_index Rebuttal_sentence Related_to Relation").split()

def build_csv_lines(review_text, rebuttal_text):
    assert len(review_text) == len(rebuttal_text)
    lines = []
    for ((review_i, review_sentence),
            (rebuttal_i, rebuttal_sentence)) in zip(review_text, rebuttal_text):
        lines.append((review_i, review_sentence, "", "", rebuttal_i,
            rebuttal_sentence, "", ""))
    return lines


def main():
    with open(sys.argv[1], 'r') as f:
        obj = json.load(f)
        pairs = obj["review_rebuttal_pairs"][:10]
        os.makedirs("pre_pilot/")
        for pair in pairs:
            review_text = get_text_as_string(pair["review_text"],
                    "review")
            rebuttal_text = get_text_as_string(pair["rebuttal_text"],
                    "rebuttal")
            max_len = max([len(rebuttal_text), len(review_text)])
            padded_review_text = pad(review_text, max_len)
            padded_rebuttal_text = pad(rebuttal_text, max_len)
            csv_lines = build_csv_lines(padded_review_text,
                    padded_rebuttal_text)

            with open(
                    "pre_pilot/template_" + str(pair["index"]) + ".csv",
                    'w') as f:
                writer = csv.writer(f)
                writer.writerow(FIELD_NAMES)
                for line in csv_lines:
                    writer.writerow(line)
                            
if __name__ == "__main__":
    main()
