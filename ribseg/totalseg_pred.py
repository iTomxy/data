import os, tempfile, glob
import numpy as np
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

"""
Test pretrained TotalSegmentator on Ribsegv2
"""

def predict(nii_file, save_file, subtask, overwrite=False):
    if not os.path.isfile(save_file) or overwrite:
        assert os.path.isfile(nii_file), nii_file
        os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_nib = totalsegmentator(nii_file, temp_dir, task=subtask)
        nib.save(pred_nib, save_file)


if "__main__" == __name__: # must, to avoid multiprocessing error
    # P = os.path.expanduser("~/data/ribseg/pt_preproc")
    P = os.path.expanduser("~/codes/tmp.ptcloud/data/ribsegv2/image")
    save_dir = os.path.expanduser("~/data/ribseg/totalseg_pred/raw")
    os.makedirs(save_dir, exist_ok=True)
    # vids = [501, 507, 570, 589, 630, 653] # heavily noisy vols
    subtask = "total"
    for vid in range(1, 660):
        if os.path.isfile(os.path.join(P, "RibFrac{}-image.nii.gz".format(vid))):
            predict(
                # os.path.join(P, "{}-image.nii.gz".format(vid)),
                os.path.join(P, "RibFrac{}-image.nii.gz".format(vid)),
                os.path.join(save_dir, "{}-ts_pred-{}.nii.gz".format(vid, subtask)),
                subtask,
            )
            print(vid, end='\r')
