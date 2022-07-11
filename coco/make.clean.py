import os.path as osp
import numpy as np
import scipy.io as sio


"""remove the data with empty label"""


COCO_P = "/home/dataset/COCO"

labels = sio.loadmat(osp.join(COCO_P, "labels.COCO.mat"))["labels"]
print(labels.shape, labels.dtype) # (123287, 80), dtype('uint8')
row_sum = labels.sum(1)
indices = np.arange(labels.shape[0])
clean_id = indices[row_sum > 0]
print(clean_id.shape)  # (122218,)
sio.savemat(osp.join(COCO_P, "clean_id.COCO.mat"), {"clean_id": clean_id})
