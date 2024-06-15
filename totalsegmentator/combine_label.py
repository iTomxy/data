import os, os.path as osp, json
import nibabel as nib

"""
Combine original separate binary segmentation annotation files into a single label volume.
See ./classes.json for correspondance of class IDs and names.
Download & extract data using ./download.sh before executing this file.
"""

with open("classes.json", "r") as f:
    idc = json.load(f)
cls_id = {} # class name -> id
for cid, cn in idc.items():
    cls_id[cn] = int(cid)


P = "/home/ftao/Data/tyliang/data/totalsegmentator"
for vid in os.listdir(osp.join(P, "data")):
    print(vid, end='\n')
    label_path = osp.join(P, "data", vid, "segmentations")
    comb_label = None
    for lab_f in os.listdir(label_path):
        cn = lab_f[:-len(".nii.gz")]
        cid = cls_id[cn]
        assert cn in cls_id, cn
        lab_nib = nib.load(osp.join(label_path, lab_f))
        lab = lab_nib.get_fdata() # float64, in {0.0, 1.0}
        if comb_label is None:
            # the original dtype is also uint8, which is enough for 117 classes
            comb_label = np.zeros_like(lab, dtype=np.uint8)

        comb_label[lab > 0.5] = cid

    # keep original affine & header (incl. spacing)
    comb_lab_nib = lab_nib.__class__(comb_label, lab_nib.affine, lab_nib.header)
    assert comb_lab_nib.shape == lab_nib.shape
    nib.save(comb_lab_nib, osp.join(osp.join(P, "data", vid, "comb_label.nii.gz")))
