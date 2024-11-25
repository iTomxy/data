import os, json
import numpy as np
import nibabel as nib

"""
Record the class set of each volume.
"""

P = os.path.expanduser("~/data/totalsegmentator")
record = {}
for vid in os.listdir(os.path.join(P, "data")):
    assert vid.startswith("s"), vid
    label = nib.load(os.path.join(P, "data", vid, "comb_label.nii.gz")).get_fdata().astype(np.int32)
    record[vid] = np.unique(label).tolist()
    print(vid, end='\r')

with open(os.path.join(P, "cls-set-per-vol.json"), "w") as f:
    json.dump(record, f)
