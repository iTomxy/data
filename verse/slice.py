import os, os.path as osp
import numpy as np
import medpy.io as medio

"""
Slice preprocessed .npy files by ./preprocess.py (and ./complete_label.py).
"""

def slice_nii(src_dir, dest_dir, slice_axis):
    """
    src_dir: str, path to (subset directories of) processed nii
    dest_dir: str, path to save slices
    slice_axis: int, in {0, 1, 2}, along which axis to slice
    """
    for subset in os.listdir(src_dir):
        print('\t', subset)
        subset_d = osp.join(src_dir, subset)
        for f in os.listdir(subset_d):
            print(f, end='\r')
            # sub-verse012_ct.nii.gz, sub-verse012_seg-vert_msk.nii.gz, sub-verse012_label_ts.nii.gz
            stem = f[:-7]
            slice_p = osp.join(dest_dir, subset, stem)
            os.makedirs(slice_p, exist_ok=True)
            # vol = np.load(osp.join(subset_d, f))
            vol, _ = medio.load(osp.join(subset_d, f))
            if stem.endswith("_msk") or stem.endswith("_label_ts"): # label, use int
                vol = vol.astype(np.uint8) # class id in [0, 28]
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


P = osp.expanduser("~/sd10t/verse")
slice_nii(osp.join(P, "processed-verse19"), osp.join(P, "processed-verse19-h-np"), 2)
slice_nii(osp.join(P, "processed-verse19-ts-bone-label"), osp.join(P, "processed-verse19-ts-bone-label-h-np"), 2)
