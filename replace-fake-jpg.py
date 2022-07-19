import argparse
import os
import os.path as osp
import cv2
from PIL import Image
import numpy as np


"""replace the fake jpeg detected by `check_non_jpg.m`
Usage
1. Before: run `check_non_jpg.m` to find out those fake jpeg files first.
2. After: re-run `check_non_jpg.m` for validation.
"""


parser = argparse.ArgumentParser(description='replace fake jpeg files')
parser.add_argument('-p', type=str, help="image path (prefix)", default="")
parser.add_argument('-f', type=str, help="txt file of fake jpeg list",
    default="fake-jpg.txt")
args = parser.parse_args()

with open(args.f, "r") as f:
    for line in f:
        line = line.strip()
        if "" == line:
            continue

        _base_f = osp.basename(line)
        if not ".jpg" in _base_f and not ".jpeg" in _base_f:
            continue

        img_p = osp.join(args.p, line)
        img = cv2.imread(img_p)#[:, :, ::-1]
        if img is None:
            with Image.open(img_p) as _img_f:
                img = np.asarray(_img_f)
            if 2 == img.ndim:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # remove old file/soft-link
        os.remove(img_p)
        # replace
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_p, img_bgr)
print("DONE")

"""fake list
--- MIR-Flickr25k
--- NUS-WIDE
/home/dataset/nuswide/Flickr/albatross/0213_10341804.jpg
/home/dataset/nuswide/Flickr/athlete/0337_2562821687.jpg
/home/dataset/nuswide/Flickr/bicycles/0544_2545048982.jpg
/home/dataset/nuswide/Flickr/dust/0084_2537626113.jpg
--- COCO
/home/dataset/COCO/train2017/000000320612.jpg
"""
