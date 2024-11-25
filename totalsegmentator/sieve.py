import os, os.path as osp, json, pprint
import numpy as np
import nibabel as nib

"""
"""

def binarise_coi(comb_lab, coi):
    """binarise class of interest
    comb_label: numpy.ndarray, [H, W, L]
    coi: List[int], classes of interest
    """
    comb_lab = comb_lab.copy().astype(np.int32) # signed, large enough
    for c in np.unique(comb_lab):
        if c in coi:
            comb_lab[c == comb_lab] = -1
        else:
            comb_lab[c == comb_lab] = -2

    bin_coi = comb_lab + 2 # {-2, -1} -> {0, 1}
    return bin_coi.astype(np.uint8)


def sieve_n_slice(src_path, dest_path, coi, c_drop=[]):
    """sieve volume slices that contain classes of interest, and slice them along the IS-axis
    src_path: str, path to original TotalSegmentator volume directories
    dest_path: str, path to save sieved & sliced volumes
    coi: List[int], class IDs of interest
    c_drop: List[int], if provided, exclude slices containing these classes
    """
    for vid in os.listdir(src_path):
        save_dir = osp.join(dest_path, vid)
        if osp.isdir(save_dir):
            continue
        print(vid, end='\r')

        comb_lab_nib = nib.load(osp.join(src_path, vid, "comb_label.nii.gz"))
        assert ('R', 'A', 'S') == nib.aff2axcodes(comb_lab_nib.affine)
        comb_lab = np.asanyarray(comb_lab_nib.dataobj) # previous `.get_data()`
        comb_lab = comb_lab.astype(np.uint8) # uint8 is enough, see ./classes.json
        assert comb_lab.shape == comb_lab_nib.shape
        bin_coi = binarise_coi(comb_lab, coi) # in {0, 1}
        if bin_coi.sum() == 0:
            continue

        poi = bin_coi > 0 # positions of interest
        poi_1d = poi.sum((0, 1)) > 0 # along axis-2, so sum axis-0 & axis-1
        if len(c_drop) > 0:
            bin_cd = binarise_coi(comb_lab, c_drop)
            cd_1d = bin_cd.sum((0, 1)) > 0
            poi_1d &= ~ cd_1d # exclude slides that contain classes to drop

        # supp = np.where(poi)
        # l, r = supp[0].min(), supp[0].max()
        # f, b = supp[1].min(), supp[1].max()
        # u, d = supp[2].min(), supp[2].max()
        # l, r = max(0, l - 4), min(r + 4, comb_lab.shape[0])
        # f, b = max(0, f - 4), min(b + 4, comb_lab.shape[1])
        # u, d = max(0, u - 4), min(d + 4, comb_lab.shape[2])

        tmp_dir = save_dir + '_tmp'
        os.makedirs(tmp_dir, exist_ok=True)

        # cl_cut = comb_lab_nib.slicer[:, :, u: d]
        cl_cut_np = comb_lab[:, :, poi_1d] # convert to np to apply fancy slicing (i.e. array indexing)
        assert cl_cut_np.shape[2] == poi_1d.sum(), f"{cl_cut_np.shape} v.s. {poi_1d.sum()}"
        cl_cut = comb_lab_nib.__class__(cl_cut_np, comb_lab_nib.affine, comb_lab_nib.header, comb_lab_nib.extra)
        assert cl_cut.shape == cl_cut_np.shape
        nib.save(cl_cut, osp.join(tmp_dir, "comb_label.nii.gz"))

        img_nib = nib.load(osp.join(src_path, vid, "ct.nii.gz"))
        assert ('R', 'A', 'S') == nib.aff2axcodes(img_nib.affine)
        img_np = img_nib.get_fdata().astype(np.float16)
        # img_cut = img_nib.slicer[:, :, u: d]
        img_cut_np = img_np[:, :, poi_1d]
        img_cut = img_nib.__class__(img_cut_np, img_nib.affine, img_nib.header, img_nib.extra)
        nib.save(img_cut, osp.join(tmp_dir, "ct.nii.gz"))

        slice_dir = osp.join(tmp_dir, "slices_is")
        if not osp.isdir(slice_dir):
            tmp_slice_dir = slice_dir + "_tmp"
            os.makedirs(tmp_slice_dir, exist_ok=True)
            for i in range(img_cut_np.shape[2]): # along IS-axis
                np.savez_compressed(osp.join(tmp_slice_dir, str(i)), image=img_cut_np[:, :, i], label=cl_cut_np[:, :, i])

            os.rename(tmp_slice_dir, slice_dir) # mark as done

        os.rename(tmp_dir, save_dir) # mark as done


P = osp.expanduser("~/sd10t/totalsegmentator")

print("pelvic, mimic ctpelvic1k")
coi = [25, 77, 78]
sieve_n_slice(osp.join(P, "data"), osp.join(P, "pelvic"), coi)

print("spine, mimic verse19")
coi = list(range(26, 50+1))
c2drop = [25, 77, 78]
sieve_n_slice(osp.join(P, "data"), osp.join(P, "spine"), coi, c2drop)

print("spine (lumbar & sacral) + pelvic (sacrum, hip)")
coi = list(range(26, 31+1)) + [25, 77, 78]
sieve_n_slice(osp.join(P, "data"), osp.join(P, "spineLSpelvic"), coi)

print("C1-7, humerus, scapula, clavicula")
coi = list(range(44, 50+1)) + [69, 70] + [71, 72] + [73, 74]
sieve_n_slice(osp.join(P, "data"), osp.join(P, "spineCshoulder"), coi)
