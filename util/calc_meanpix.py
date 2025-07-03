import numpy as np
import os.path as osp
import scipy.io as sio
from util.loaders import ImageF25k, ImageNUS, ImageCOCO


"""Precompute the mean pixel
The matlab version is implemented within (Gitee) tyloeng/reimpl.DCMH (matlab branch).
"""


def calc_meanpix(images, image_size=224, batch_size=128):
    n = len(images)
    meanpix = np.zeros((image_size, image_size, 3), dtype=np.float32)
    indices = np.arange(n)
    for i in range(0, n, batch_size):
        idx = indices[i: i + batch_size]
        img = images[idx]
        meanpix += img.astype(np.float32).sum(0)
        print(i)

    meanpix /= n
    print(meanpix.mean(), meanpix.min(), meanpix.max())
    return meanpix


DATA_ROOT = "/usr/local/dataset"

F25K_P = osp.join(DATA_ROOT, "flickr25k")
mean_f25k = calc_meanpix(ImageF25k(osp.join(F25K_P, "mirflickr")))
# (224, 224, 3) 103.97862 70.17312 130.13568
sio.savemat(osp.join(F25K_P, "avgpix.flickr25k.py.mat"), {"meanpix": mean_f25k})

NUS_P = osp.join(DATA_ROOT, "nuswide")
mean_nus = calc_meanpix(ImageNUS(osp.join(NUS_P, "images")))
# (224, 224, 3) 106.57295352731454 76.35340888862517 130.3679760280069
sio.savemat(osp.join(NUS_P, "avgpix.nuswide.py.mat"), {"meanpix": mean_nus})

COCO_P = osp.join(DATA_ROOT, "COCO")
mean_coco = calc_meanpix(ImageCOCO(osp.join(COCO_P, "images")))
# (224, 224, 3) 112.46002921296414 87.87884367370445 131.3540357053055
sio.savemat(osp.join(COCO_P, "avgpix.COCO.py.mat"), {"meanpix": mean_coco})
