import os, json
from collections import defaultdict
import numpy as np
# from scipy.ndimage import zoom
from dipy.align.reslice import reslice
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation

"""
Preprocess CT scans & labels to unify orientation & spacing.

Prerequisites:
    - Gather images into one folder using ./link-data.sh.
"""

def inspect_meta(image_path):
    """Check the statistics of original nii meta information
    image_path: str, path to the unified folder where image nii files are gathered by ./link-data.sh.
    """
    ori_set = defaultdict(lambda: 0)
    spacing_stat = [[] for _ in range(3)]
    with open("volume-meta.json", "w") as log_f:
        for f in sorted(os.listdir(image_path), key=lambda _f:  int(_f.split('-')[0][7:])):
            vid = int(f.split('-')[0][7:])
            ct_nii = nib.load(osp.join(image_path, f))
            ori = ''.join(nib.aff2axcodes(ct_nii.affine))
            spc = np.asarray(ct_nii.header.get_zooms()).tolist()
            print(vid, ori, spc, end='\r')
            log_f.write(json.dumps({
                "vid": vid,
                "orientation": ori,
                "spacing": spc
            }) + os.linesep)

            ori_set[ori] += 1
            for i, s in enumerate(spc):
                spacing_stat[i].append(s)

    print(ori_set) # {'LPS': 659, 'RAS': 1}
    for ax, v_list in enumerate(spacing_stat):
        # 0 0.556640625 0.9765629768371582 0.7400686852859728 0.734375
        # 1 0.556640625 0.9765629768371582 0.7400686852859728 0.734375
        # 2 0.625 2.0 1.1327273050944011 1.25
        print(ax, np.min(v_list), np.max(v_list), np.mean(v_list), np.median(v_list))


def unify(image_nii, label_nii, target_orientation=('L', 'P', 'S'), target_spacing=(0.6, 0.6, 0.6)):
    """unify orientation and spacing"""
    original_orientation = nib.aff2axcodes(image_nii.affine)
    original_spacing = np.array(image_nii.header.get_zooms())[:3]
    # print(f"Original orientation: {original_orientation}")
    # print(f"Original spacing: {original_spacing}")

    # Step 1: Reorient to target orientation if not already
    if original_orientation != target_orientation:
        # print(f"Reorienting from {original_orientation} to {target_orientation}...")

        # Convert orientation strings to orientation matrices
        orig_ornt = axcodes2ornt(original_orientation)
        target_ornt = axcodes2ornt(target_orientation)

        # Get the transform from original to target orientation
        transform = ornt_transform(orig_ornt, target_ornt)

        # Apply the orientation transform to the image
        image_data = apply_orientation(image_nii.get_fdata(), transform)

        # Create new affine for the transformed image
        affine = image_nii.affine.copy()
        affine = nib.orientations.inv_ornt_aff(transform, image_nii.shape)
        affine = np.dot(image_nii.affine, affine)

        # Create new oriented image
        image_nii = image_nii.__class__(image_data, affine, image_nii.header)

        # Apply the same orientation transform to the label
        label_data = apply_orientation(label_nii.get_fdata(), transform)

        # Create new oriented label
        label_affine = label_nii.affine.copy()
        label_affine = nib.orientations.inv_ornt_aff(transform, label_nii.shape)
        label_affine = np.dot(label_nii.affine, label_affine)
        label_nii = label_nii.__class__(label_data.astype(np.uint8), label_affine, label_nii.header)

    # Check orientation after reorientation
    final_orientation = nib.aff2axcodes(image_nii.affine)
    # print(f"Orientation after reorientation: {final_orientation}")
    assert nib.aff2axcodes(image_nii.affine) == target_orientation

    # Step 2: Resample to target spacing
    # print(f"Resampling to spacing: {target_spacing}")

    # Get current spacing
    current_spacing = np.array(image_nii.header.get_zooms())[:3]

    # Resample the CT scan
    new_image_data, new_image_affine = reslice(image_nii.get_fdata(), image_nii.affine,
                                              current_spacing, target_spacing)
    resampled_ct = image_nii.__class__(new_image_data.astype(np.float16), new_image_affine, image_nii.header)

    # Update the header's zoom fields to match the new spacing
    resampled_ct.header.set_zooms(target_spacing)

    # Resample label using nearest neighbor interpolation
    new_label_data, new_label_affine = reslice(label_nii.get_fdata().astype(np.uint8), label_nii.affine,
                                              current_spacing, target_spacing, order=0)
    resampled_label = label_nii.__class__(new_label_data.astype(np.uint8), new_label_affine, label_nii.header)
    resampled_label.header.set_zooms(target_spacing)

    # Verify the new spacing
    # new_spacing = np.array(resampled_ct.header.get_zooms())[:3]
    # print(f"New spacing: {new_spacing}")

    return resampled_ct, resampled_label


if "__main__" == __name__:
    data_base = os.path.expanduser("~/codes/tmp.ptcloud/ribsegv2/segmentation/data/ribsegv2")
    image_path = os.path.join(data_base, "image")
    label_path = os.path.join(data_base, "label")

    print("inspect meta info")
    # inspect_meta(image_path)

    print("preprocessing: unify orientation & spacing")
    target_orientation = ('L', 'P', 'S')
    target_spacing = (0.6, 0.6, 0.6)
    save_base = os.path.expanduser("~/data/ribseg/v2-uni-o{}-s{}".format(
        ''.join(target_orientation),
        '_'.join(map(str, target_spacing))
    ))
    save_image_dir = os.path.join(save_base, "image")
    save_label_dir = os.path.join(save_base, "label")
    for d in (save_image_dir, save_label_dir):
        os.makedirs(d, exist_ok=True)

    for src_img_f in os.listdir(image_path):
        vid = int(src_img_f.split('-')[0][7:]) # RibFrac<VID>-image.nii.gz
        # print(vid, end='\r')
        src_lab_f = src_img_f.replace("-image", "-rib-seg") # RibFrac<VID>-rig-seg.nii.gz

        dest_img_f = os.path.join(save_image_dir, src_img_f)
        dest_lab_f = os.path.join(save_label_dir, src_lab_f)
        if os.path.isfile(dest_img_f) and os.path.isfile(dest_lab_f):
            continue

        image_nii = nib.load(os.path.join(image_path, src_img_f))
        label_nii = nib.load(os.path.join(label_path, src_lab_f))

        new_image_nii, new_label_nii = unify(image_nii, label_nii, target_orientation, target_spacing)
        # print(image_nii.shape, new_image_nii.shape)
        # print(image_nii.header.get_zooms(), new_image_nii.header.get_zooms())
        # print(nib.aff2axcodes(image_nii.affine), nib.aff2axcodes(new_image_nii.affine))

        nib.save(new_image_nii, dest_img_f)
        nib.save(new_label_nii, dest_lab_f)
        print(vid, nib.aff2axcodes(new_image_nii.affine), new_image_nii.header.get_zooms(), nib.aff2axcodes(new_label_nii.affine), new_label_nii.header.get_zooms(), end='\r')
