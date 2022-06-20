import os
import os.path as osp
import numpy as np
import scipy.io as sio


"""data cleaning
Filter out those with no tags or labels.
"""


P = "G:/flickr-25k.origin"
labels = sio.loadmat(osp.join(P, "labels.flickr25k.mat"))["labels"]
texts = sio.loadmat(osp.join(P, "texts.flickr25k.mat"))["texts"]
print(labels.shape, texts.shape)

id_clean = []
for sid in range(labels.shape[0]):
    if (texts[sid].sum() > 0) and (labels[sid].sum() > 0):
        id_clean.append(sid)
id_clean = np.asarray(id_clean)
print("#clean sample:", id_clean.shape, id_clean.dtype)  # (20015,) int32
sio.savemat(osp.join(P, "clean_id.flickr25k.mat"), {"clean_id": id_clean})
