import os, os.path as osp
import numpy as np


"""
Slice preprocessed .npy files by ./preprocess.py.
"""


P = osp.expanduser("~/data/verse")
SAVE_P = osp.join(P, "processed-verse19", "npy")
SLICE_P = osp.join(P, "processed-verse19-npy-horizontal")
# os.makedirs(SLICE_P, exist_ok=True)
AXIS = 2 # RAI, horizontal

for subset in os.listdir(SAVE_P):
    print('\t', subset)
    subset_d = osp.join(SAVE_P, subset)
    for f in os.listdir(subset_d):
        print(f, end='\r')
        stem = osp.splitext(f)[0]
        slice_p = osp.join(SLICE_P, subset, stem)
        os.makedirs(slice_p, exist_ok=True)
        vol = np.load(osp.join(subset_d, f))
        # if osp.isdir(slice_p):
        #     if os.listdir(slice_p) == vol.shape[AXIS]: # skip finished
        #         continue
        # else:
        #     os.makedirs(slice_p)#, exist_ok=True)
        for i in range(vol.shape[AXIS]):
            # if osp.isfile(osp.join(slice_p, f"{stem}_{i}.npy")): # skip finished
            #     continue
            print(i, end='\r')
            s = vol[:, :, i]
            np.save(osp.join(slice_p, f"{stem}_{i}"), s)
