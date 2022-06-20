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
parser.add_argument('-p', type=str, help="image path",
    default="/home/dataset/nuswide/images")
parser.add_argument('-f', type=str, help="txt file of fake jpeg list",
    default="fake-jpg.txt")
args = parser.parse_args()

with open(args.f, "r") as f:
    for line in f:
        img_p = osp.join(args.p, line.strip())
        img = cv2.imread(img_p)#[:, :, ::-1]
        if img is None:
            with Image.open(img_p) as _img_f:
                img = np.asarray(_img_f)
            if 2 == img.ndim:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # remove old link
        os.remove(img_p)
        # replace
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_p, img_bgr)
print("DONE")
