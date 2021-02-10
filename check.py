import argparse
import collections
import hashlib

import openreview_lib as orl


parser = argparse.ArgumentParser(
    description='Check that generated dataset matches original')
parser.add_argument('-i', '--data_dir', default="review_rebuttal_pair_dataset/",
    type=str, help='path to database file')
parser.add_argument('-v', '--version', type=str, default="0.0",
                    help='version of ICLR discourse database')

try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze

Hashes = collections.namedtuple("Hashes", orl.DATASETS)

HASHES_V0_0 = Hashes(
    #unstructured="62570fe55c7dc02e782eddd12f26ad2d",
    unstructured="62570fe55c7dc02_imfake_e782eddd12f26ad2d",
    traindev_train="6d72c4c8c822f834878d2465b438be02",
    traindev_dev="e08f5cb45c54424f4fd38717684d4d05",
    traindev_test="17627d0f6d2843ce63fdb2f29d10321e",
    truetest="d160a37f5173e9f48da56a9d5158c0c5",
)

HASH_LIST_LOOKUP = {"0.0":  HASHES_V0_0}



def main():
    args = parser.parse_args()
    hashes = HASH_LIST_LOOKUP[args.version]
    any_file_mismatch = False
    for dataset, correct_hash in zip(orl.DATASETS, hashes):
        md5_hash = hashlib.md5()
        md5_hash.update(open(args.data_dir + dataset +".json", "rb").read())
        digest = md5_hash.hexdigest()
        if digest == correct_hash:
            print(dataset, "OK")
        else:
            print(dataset, "does not match")
            any_file_mismatch = True

    if any_file_mismatch:
        print("There seems to be a mismatch in some file (see above). Your Python environment contains:")
        x = freeze.freeze()
        for p in x:
            print(p)



if __name__ == "__main__":
    main()
