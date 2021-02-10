import hashlib

DATASET_FILE_TO_HASH = {
    "unstructured.json": "62570fe55c7dc02e782eddd12f26ad2d",
    "traindev_train.json": "6d72c4c8c822f834878d2465b438be02",
    "traindev_dev.json": "e08f5cb45c54424f4fd38717684d4d05",
    "traindev_test.json": "17627d0f6d2843ce63fdb2f29d10321e",
    "truetest.json": "d160a37f5173e9f48da56a9d5158c0c5"
}


def main():
    directory = "review_rebuttal_pair_dataset/"
    for dataset_file, hash_value in DATASET_FILE_TO_HASH.items():
        md5_hash = hashlib.md5()
        md5_hash.update(open(directory + dataset_file, "rb").read())
        digest = md5_hash.hexdigest()
        if digest == hash_value:
            print(dataset_file, "OK")
        else:
            print(dataset_file, "does not match")



if __name__ == "__main__":
    main()
