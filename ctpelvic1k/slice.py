import os, os.path as osp, glob
import numpy as np
import medpy.io as medio

"""
Slice preprocessed .npy files by ./preprocess.py and ./complete_label.py.
- (2024.6.21) Compress image, label & TS label together in an npz file.
"""

SLICE_AXIS = 2 # RAI, slice along the IS-axis

P = osp.expanduser("~/sd10t/ctpelvic1k")
il_path = osp.join(P, "processed-ctpelvic1k") # image & label
tslab_path = osp.join(P, "processed-ctpelvic1k-ts-bone-label") # TS label
save_path = osp.join(P, "processed-ctpelvic1k-slice-is")

for img_f in glob.glob(osp.join(il_path, "*_image.nii.gz")):
    vid = osp.basename(img_f)[: - len("_image.nii.gz")]
    print('\t', vid, end='\r')

    lab_f = osp.join(il_path, f"{vid}_label.nii.gz")
    tslab_f = osp.join(tslab_path, f"{vid}_label_ts.nii.gz")

    slice_p = osp.join(save_path, vid)
    if osp.isdir(slice_p):
        continue

    tmp_slice_p = slice_p + "_tmp"
    os.makedirs(tmp_slice_p, exist_ok=True)

    img, _ = medio.load(img_f)
    lab, _ = medio.load(lab_f)
    tslab, _ = medio.load(tslab_f)
    assert img.shape == lab.shape and img.shape == tslab.shape

    lab = lab.astype(np.uint8) # class id in [0, 4]
    tslab = tslab.astype(np.uint8) # class id in {0, 1}

    if 0 != SLICE_AXIS:
        img = np.moveaxis(img, SLICE_AXIS, 0)
        lab = np.moveaxis(lab, SLICE_AXIS, 0)
        tslab = np.moveaxis(tslab, SLICE_AXIS, 0)

    for i in range(img.shape[0]):
        np.savez_compressed(osp.join(tmp_slice_p, str(i)), image=img[i], label=lab[i], ts_label=tslab[i])
        print(i, end='\r')

    os.rename(tmp_slice_p, slice_p)
