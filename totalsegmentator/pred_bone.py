import os, tempfile
import numpy as np
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

"""
Find UNLABELLED bones with pretrained TotalSegmentator model.
The released dataset contains labels of the `total` task.
But some bones are not labelled.
This script finds them via the pretrained model in subtask `appendicular_bones` and `vertebrae_body`.
See: https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#subtasks.

Potential multiprocessing error:
- https://github.com/wasserth/TotalSegmentator/issues/190
- https://github.com/wasserth/TotalSegmentator/issues/198
"""

if "__main__" == __name__: # must, to avoid multiprocessing error
    P = os.path.expanduser("~/data/totalsegmentator")
    for vid in os.listdir(os.path.join(P, "data")):
        assert vid.startswith('s'), vid
        v_dir = os.path.join(P, "data", vid)
        for subtask in ("appendicular_bones", "vertebrae_body"):
            save_f = os.path.join(v_dir, f"ts_pred-{subtask}.nii.gz")
            if not os.path.isfile(save_f):
                with tempfile.TemporaryDirectory() as temp_dir:
                    pred_nib = totalsegmentator(os.path.join(v_dir, "ct.nii.gz"), temp_dir, task=subtask)
                nib.save(pred_nib, save_f)
                print(vid, subtask, end='\r')
