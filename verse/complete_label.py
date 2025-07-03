import argparse, os, os.path as osp, glob, tempfile
import importlib.metadata#, packaging
import numpy as np
import nibabel as nib
import totalsegmentator as totalseg
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.python_api import totalsegmentator

"""
Infer with pretrained TotalSegmentator models to find potential unlabelled bones.
"""

TOTALSEG_VERSION = importlib.metadata.version('totalsegmentator')
MAJOR_VERSION = int(TOTALSEG_VERSION.split('.')[0])
assert MAJOR_VERSION in (1, 2), TOTALSEG_VERSION


def is_bone(pred_np):
    """Class list of TotalSegmentator: totalsegmentator.map_to_binary.class_map
    Link: https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py
    """
    # print(class_map['total_v1'])
    if 1 == MAJOR_VERSION:
        return ((18 <= pred_np) & (pred_np <= 41)) | ((58 <= pred_np) & (pred_np <= 92))
    else:
        return ((25 <= pred_np) & (pred_np <= 50)) | ((69 <= pred_np) & (pred_np <= 78)) + ((91 <= pred_np) & (pred_np <= 116))


# def complete(image_nii, label_nii, save_nii):
#     pred_nib = totalsegmentator(image_nii, './_ts_output')
#     shutil.rmtree('./_ts_output') # rm unused results
#     label_nib = nib.load(label_nii)
#     assert pred_nib.shape == label_nib.shape, f"label: {label_nib.shape}, prediction: {pred_nib.shape}"
#     label = label_nib.get_fdata().astype(np.uint8) # in [0, 4]
#     pred = pred_nib.get_fdata().astype(np.uint8) # in [0, 104]
#     # combine original partial label & TotalSegmentator prediction
#     comb_isb = ((label > 0) | is_bone(pred)).astype(np.uint8)
#     isb_nib = nib.Nifti1Image(comb_isb, affine=label_nib._affine, header=label_nib.header)
#     os.makedirs(osp.dirname(save_nii) or '.', exist_ok=True)
#     nib.save(isb_nib, save_nii)

#     # print("test shape consistency")
#     # isb_nib = nib.load(save_nii)
#     # assert label.shape == isb_nib.shape, f"label: {label_nib.shape}, re-read: {isb_nib.shape}"


def predict(image_nii_f, vid, save_dir, subtasks=("total", "appendicular_bones", "vertebrae_body")):
    """predict with pretrained model of several bone-related subtasks, and save"""
    for subtask in subtasks:
        save_f = os.path.join(save_dir, f"{vid}-{subtask}.nii.gz")
        if not os.path.isfile(save_f):
            with tempfile.TemporaryDirectory() as temp_dir:
                pred_nib = totalsegmentator(image_nii_f, temp_dir, task=subtask)
            nib.save(pred_nib, save_f)
            print(vid, subtask, end='\r')


def combine():
    """combine given label and prediction."""
    pass


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="path to original image & label nii file")
    parser.add_argument("dest", type=str, help="path to save combined bone label nii")
    args = parser.parse_args()

    assert osp.isdir(args.src), args.src
    os.makedirs(args.dest, exist_ok=True)
    for img_f in glob.glob(osp.join(args.src, "*_image.nii.gz")):
        lab_f = img_f[:-len("_image.nii.gz")] + "_label.nii.gz"
        assert osp.isfile(lab_f), f"image: {img_f}, label: {lab_f}"
        vid = osp.basename(img_f)[:-len("_image.nii.gz")]
        print(vid, end='\r')
        # save_f = osp.join(args.dest, stem + "_label_ts.nii.gz")
        # complete(img_f, lab_f, vid)
        predict(img_f, vid, args.dest)
