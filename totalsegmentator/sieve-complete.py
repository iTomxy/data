import os, json, time, random
import numpy as np
import nibabel as nib

"""
Sieve volumes that has NO UNLABELLED BONES based on pretrained TotalSegmentator model.
(see: ./pred_bone.py)
Volumes are treated as completely labelled if:
- `appendicular_bones` prediction is empty.
"""

if "__main__" == __name__:
    print("count prediction cardinality")
    P = os.path.expanduser("~/data/totalsegmentator/data")
    card_list = {}
    for vid in os.listdir(P):
        print(vid, end='\r')
        pred_f = os.path.join(P, vid, "ts_pred-appendicular_bones.nii.gz")
        pred = nib.load(pred_f).get_fdata().astype(np.int32)
        card_list[vid] = int(pred.sum())

    with open("ts-pred-cardinality.json", "w") as f:
        json.dump({
            "time": time.asctime(),
            "appendicular_bones": card_list,
        }, f)


    print("sieve clean volumes by prediction cardinality")
    # with open("ts-pred-cardinality.json", "r") as f:
    #     card_list = json.load(f)["appendicular_bones"]
    NOISE_TOLERANCE = 100 # in case there is noise
    complete_vols = [v for v, c in card_list.items() if c <= NOISE_TOLERANCE]
    print("#complete volumes:", len(complete_vols))
    with open("complete-volumes.json", "w") as f:
        json.dump({
            "time": time.asctime(),
            "noise_tolerance": NOISE_TOLERANCE,
            "volume": complete_vols,
        }, f)
