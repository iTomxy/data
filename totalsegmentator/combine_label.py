import os, os.path as osp, json, pprint
import numpy as np
import nibabel as nib
from map_to_binary import class_map

"""
Combine original separate binary segmentation annotation files into a single label volume.
See ./map_to_binary.py for correspondance of class IDs and names.
Download & extract data using ./download.sh before executing this file.
"""


def combine_total(data_root=osp.expanduser("~/data/totalsegmentator")):
    """combine class-wise binary label volumes (belonging to the `total` task) into a single label volume"""
    # class name -> id
    cls_id = {cn: cid for cid, cn in class_map["total"].items()}
    for vid in os.listdir(osp.join(data_root, "data")):
        print(vid, end='\r')
        label_path = osp.join(data_root, "data", vid, "segmentations")
        comb_label = None
        for lab_f in os.listdir(label_path):
            cn = lab_f[:-len(".nii.gz")]
            cid = cls_id[cn]
            assert cn in cls_id, cn
            lab_nib = nib.load(osp.join(label_path, lab_f))
            assert ('R', 'A', 'S') == nib.aff2axcodes(lab_nib.affine) # confirm consistent orientation
            lab = lab_nib.get_fdata() # float64, in {0.0, 1.0}
            if comb_label is None:
                # the original dtype is also uint8, which is enough for 117 classes
                comb_label = np.zeros_like(lab, dtype=np.uint8)

            comb_label[lab > 0.5] = cid

        # keep original affine & header (incl. spacing)
        comb_lab_nib = lab_nib.__class__(comb_label, lab_nib.affine, lab_nib.header, lab_nib.extra)
        assert comb_lab_nib.shape == lab_nib.shape
        nib.save(comb_lab_nib, osp.join(osp.join(P, "data", vid, "comb_label.nii.gz")))


def combine_total_appendicular(data_root=osp.expanduser("~/data/totalsegmentator")):
    """combine labels from `tatal` and `appendicular_bones`
    New appendicular bone cid = max{total cid} + original appendicular bone cid
    """
    max_total_cid = max(class_map["total"].keys())
    # print(max_total_cid) # 117
    for vid in os.listdir(osp.join(data_root, "data")):
        print(vid, end='\r')
        vol_dir = osp.join(data_root, "data", vid)
        img_nib = nib.load(osp.join(vol_dir, "ct.nii.gz"))
        lab_total = nib.load(osp.join(vol_dir, "comb_label.nii.gz")).get_fdata().astype(np.uint8)
        lab_app = nib.load(osp.join(vol_dir, "ts_pred-appendicular_bones.nii.gz")).get_fdata().astype(np.uint8)
        # print(np.unique(lab_total))
        # print(np.unique(lab_app))
        for app_cid in np.unique(lab_app):
            if app_cid > 0:
                # mask = lab_app == app_cid
                # print(np.unique(lab_total[mask]))
                lab_total[lab_app == app_cid] = max_total_cid + app_cid

        comb_lab_nib = img_nib.__class__(lab_total, img_nib.affine, img_nib.header, img_nib.extra)
        assert comb_lab_nib.shape == img_nib.shape
        nib.save(comb_lab_nib, osp.join(vol_dir, "comb_label_appendicular.nii.gz"))


if "__main__" == __name__:
    combine_total()
    combine_total_appendicular()
