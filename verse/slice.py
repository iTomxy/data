import os, os.path as osp
import numpy as np
import medpy.io as medio


"""
Slice preprocessed .npy files by ./preprocess.py.
"""


P = osp.expanduser("~/data/verse")
SAVE_P = osp.join(P, "processed-verse19-wl300-ww290-std")#, "npy")
SLICE_P = osp.join(P, "processed-verse19-wl300-ww290-std-h-np")
# os.makedirs(SLICE_P, exist_ok=True)
AXIS = 2 # RAI, horizontal

for subset in os.listdir(SAVE_P):
    print('\t', subset)
    subset_d = osp.join(SAVE_P, subset)
    for f in os.listdir(subset_d):
        print(f, end='\r')
        # sub-verse012_ct.nii.gz, sub-verse012_seg-vert_msk.nii.gz
        stem = f[:-7]
        slice_p = osp.join(SLICE_P, subset, stem)
        os.makedirs(slice_p, exist_ok=True)
        # vol = np.load(osp.join(subset_d, f))
        vol, _ = medio.load(osp.join(subset_d, f))
        if stem.endswith("_msk"): # label, use int
            vol = vol.astype(np.uint8) # class id in [0, 28]
        # if osp.isdir(slice_p):
        #     if os.listdir(slice_p) == vol.shape[AXIS]: # skip finished
        #         continue
        # else:
        #     os.makedirs(slice_p)#, exist_ok=True)
        if 0 != AXIS:
            vol = np.moveaxis(vol, AXIS, 0)
        for i in range(vol.shape[0]):
            # if osp.isfile(osp.join(slice_p, f"{stem}_{i}.npy")): # skip finished
            #     continue
            print(i, end='\r')
            s = vol[i]
            np.save(osp.join(slice_p, f"{stem}_{i}"), s)
