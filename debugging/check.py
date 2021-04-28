import argparse
import collections
import hashlib

import openreview_lib as orl


parser = argparse.ArgumentParser(
    description='Check that generated dataset matches original')
parser.add_argument('-i', '--data_dir', default="data/review_rebuttal_pair_dataset/",
    type=str, help='path to database file')
parser.add_argument('-v', '--version', type=str, default="0.1",
                    help='version of ICLR discourse database')
parser.add_argument('-d', '--debug', action='store_true',
                    help='truncate to small subset of data for debugging')

try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze

Hashes = collections.namedtuple("Hashes", orl.DATASETS)

HASHES_V0_0 = Hashes(
    unstructured="62570fe55c7dc02e782eddd12f26ad2d",
    traindev_train="6d72c4c8c822f834878d2465b438be02",
    traindev_dev="e08f5cb45c54424f4fd38717684d4d05",
    traindev_test="17627d0f6d2843ce63fdb2f29d10321e",
    truetest="d160a37f5173e9f48da56a9d5158c0c5",
)

HASHES_V0_0_debug = Hashes(
    unstructured="efc77f1b6abd16fbe2476cebf854d73d",
    traindev_train="d7366e017d4e4c551e3546acd695e5d7",
    traindev_dev="5db7f4277fb7f28ad6e16970a6ea94a7",
    traindev_test="05b40325b6d2bb02698119bb8896c971",
    truetest="f4e02d67a7a205205ceb0c529ed6377c",
)

HASHES_V0_1 = Hashes(
    unstructured="1b1e2b13dcd8912b001067f35ecc94dd",
    traindev_train="d22e81671c06e42a0f4e0d0666ebefbd",
    traindev_dev="e6ba473f5ce98e09bebb828f57155b43",
    traindev_test="55a85c680efa63f4a92fc83f7271f4e7",
    truetest="c37b170ccc6f4885d60d6856e2725e49",
)

HASHES_V0_1_debug = Hashes(
    unstructured="a3304183f7eb16b86df5f630f0791afa",
    traindev_train="0dc10ca048a0452a3714d06c0ab4a9b0",
    traindev_dev="5ee081bd1a70a5b5275c67139528d78b",
    traindev_test="702dbcdac732f9ecdf7d375019170e1d",
    truetest="c1e961ae8a66848c764c8dc041285c10",
)


HASH_LIST_LOOKUP = {
                    "0.0":  HASHES_V0_0,
                    "0.0_debug": HASHES_V0_0_debug,
                    "0.1":  HASHES_V0_1,
                    }



def main():
    args = parser.parse_args()

    hashes_key = args.version
    data_dir = args.data_dir
    if args.debug:
        hashes_key += "_debug"
        data_dir = args.data_dir[:-1]+"_debug/"


    hashes = HASH_LIST_LOOKUP[hashes_key]
    any_file_mismatch = False
    for dataset, correct_hash in zip(orl.DATASETS, hashes):
        md5_hash = hashlib.md5()
        md5_hash.update(open(data_dir + dataset +".json", "rb").read())
        digest = md5_hash.hexdigest()
        print(dataset, digest)
        if digest == correct_hash:
            print(dataset, "OK")
        else:
            print(dataset, "does not match")
            any_file_mismatch = True


if __name__ == "__main__":
    main()
