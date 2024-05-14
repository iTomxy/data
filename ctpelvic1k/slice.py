import os, os.path as osp
import numpy as np
import medpy.io as medio

"""
Slice preprocessed .npy files by ./preprocess.py (and ./complete_label.py).
"""

def slice_nii(src_dir, dest_dir, slice_axis):
    for f in os.listdir(src_dir):
        """
        src_dir: str, path to (subset directories of) processed nii
        dest_dir: str, path to save slices
        slice_axis: int, in {0, 1, 2}, along which axis to slice
        """
        print(f, end='\r')
        stem = f[:-7] # osp.splitext(f)[0]
        slice_p = osp.join(dest_dir, stem)
        os.makedirs(slice_p, exist_ok=True)
        # vol = np.load(osp.join(src_dir, f))
        vol, _ = medio.load(osp.join(src_dir, f))
        if stem.endswith("_label") or stem.endswith("_label_ts"): # label, use int
            vol = vol.astype(np.uint8) # class id in [0, 4]
        # if osp.isdir(slice_p):
        #     if os.listdir(slice_p) == vol.shape[slice_axis]: # skip finished
        #         continue
        # else:
        #     os.makedirs(slice_p)#, exist_ok=True)
        if 0 != slice_axis:
            vol = np.moveaxis(vol, slice_axis, 0)
        for i in range(vol.shape[0]):
            # if osp.isfile(osp.join(slice_p, f"{stem}_{i}.npy")): # skip finished
            #     continue
            print(i, end='\r')
            s = vol[i]
            np.save(osp.join(slice_p, f"{stem}_{i}"), s)


P = osp.expanduser("~/sd10t/ctpelvic1k")
slice_nii(osp.join(P, "processed-ctpelvic1k"), osp.join(P, "processed-ctpelvic1k-h-np"), 2)
slice_nii(osp.join(P, "processed-ctpelvic1k-ts-bone-label"), osp.join(P, "processed-ctpelvic1k-ts-bone-label-h-np"), 2)
