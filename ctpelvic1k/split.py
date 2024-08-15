import os, os.path as osp, glob, time, random, json


"""
Split preprocessed volumes by ./preprocess.py.
Split each sub-dataset respectively.
"""

P = osp.expanduser("~/data/ctpelvic1k")
VOLUME_P = osp.join(P, "processed-ctpelvic1k")

TRAIN_RATIO = 0.6
TEST_RATIO = 0.2
VAL_RATIO = 1 - TRAIN_RATIO - TEST_RATIO

record = {
    "time": "UTC " + time.asctime(time.gmtime()),
    "train_ratio": TRAIN_RATIO,
    "test_ratio": TEST_RATIO,
    "validation_ratio": VAL_RATIO,
    "splitting": {}
}
for sub_id in range(1, 8):
    print(sub_id, end='\r')
    vol_list = glob.glob(osp.join(VOLUME_P, f"d{sub_id}_*_image.nii.gz"))
    n = len(vol_list)
    if 0 == n:
        # record[f"d{sub_id}"] = 0 # (2023.12.6) d2 not downloaded
        continue
    n_train = int(n * TRAIN_RATIO)
    n_test = int(n * TEST_RATIO)
    n_val = n - n_train - n_test
    random.shuffle(vol_list)

    # d1_img0001_image.npy, d5_0507688_Image_label.npy
    vol_list = [osp.basename(f) for f in vol_list]
    vol_list = [f[: f.rfind('_')] for f in vol_list]

    record["splitting"][f"d{sub_id}"] = {
        "training": vol_list[:n_train],
        "test": vol_list[n_train: n_train + n_test],
        "validation": vol_list[n_train + n_test:]
    }
    # break

with open(osp.join(P, "splitting-ctpelvic1k.json"), "w") as f:
    json.dump(record, f, indent=1)
