import os, os.path as osp, glob, time, random, json

"""
Split preprocessed volumes by ./sieve.py.
Split each sub-dataset respectively.
"""

P = osp.expanduser("~/sd10t/totalsegmentator")
TRAIN_RATIO = 0.6
TEST_RATIO = 0.2
VAL_RATIO = 1 - TRAIN_RATIO - TEST_RATIO


def split(sub_dataset, small=0, complete=False):
    """
    sub_dataset: str, see ./sieve.py for sieved sub-datasets.
    small: int = 0, >0 to make a small subset of it
    complete: bool = False,
    """
    vol_list = os.listdir(osp.join(P, sub_dataset))

    if complete:
        assert os.path.isfile("complete-volumes.json"), "Run ./sieve-complete.py to select complete volumes first"
        with open("complete-volumes.json", "r") as f:
            complete_vols = json.load(f)["volume"]
        vol_list = list(set(complete_vols).intersection(set(vol_list)))

    n = len(vol_list)
    assert n > 0
    print(vol_list[:5])

    if small > 0:
        random.shuffle(vol_list)
        vol_list = vol_list[:small]
        n = len(vol_list)
    #     n_train = int(n // 2)
    #     n_test = int(n // 4)
    #     n_val = n - n_train - n_test
    # else:
    n_train = int(n * TRAIN_RATIO)
    n_test = int(n * TEST_RATIO)
    n_val = n - n_train - n_test

    random.shuffle(vol_list)
    record = {
        "time": "UTC " + time.asctime(time.gmtime()),
        "train_ratio": TRAIN_RATIO,
        "test_ratio": TEST_RATIO,
        "validation_ratio": VAL_RATIO,
        "only_complete_volumes": complete,
        "splitting": {}
    }
    record["splitting"]["training"] = vol_list[: n_train]
    record["splitting"]["test"] = vol_list[n_train: n_train + n_test]
    record["splitting"]["validation"] = vol_list[n_train + n_test: ]
    fn = f"splitting-{sub_dataset}"
    if small: fn += "-small"
    # with open(osp.join(P, f"{fn}.json"), "w") as f:
    with open(fn + ".json", "w") as f:
        json.dump(record, f)#, indent=1)


split("pelvic")
split("pelvic", 200)
split("spine")
split("spine", 200)
split("spineLSpelvic")
split("spineLSpelvic", 200)
split("spineCshoulder", 100, True)
