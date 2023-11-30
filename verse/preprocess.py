import os, os.path as osp, glob
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import itk


"""
From: https://github.com/xmed-lab/GenericSSL/blob/main/code/data/preprocess_mmwhs.py

- Use verse'19
- Use its original splitting

Data structure:
verse/
|- dataset-verse19training/
|  |- derivatives/
|  |  |- sub-verse<ID>/
|  |  |  |- sub-verse<ID>_ct.nii.gz
|  `- rawdata/
|     |- sub-verse<ID>/
|     |  |- sub-verse<ID>_seg-vert_msk.nii.gz
|- dataset-verse19validation/
`- dataset-verse19test/
"""


def read_reorient2RAI(path):
    itk_img = itk.imread(path)

    filter = itk.OrientImageFilter.New(itk_img)
    filter.UseImageDirectionOn()
    filter.SetInput(itk_img)
    m = itk.Matrix[itk.D, 3, 3]()
    m.SetIdentity()
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    itk_img = filter.GetOutput()

    itk_arr = itk.GetArrayViewFromImage(itk_img)
    return itk_arr


def getRangeImageDepth(label):
    d = np.any(label, axis=(1,2))
    h = np.any(label, axis=(0,2))
    w = np.any(label, axis=(0,1))

    if len(np.where(d)[0]) >0:
        d_s, d_e = np.where(d)[0][[0,-1]]
    else:
        d_s = d_e = 0

    if len(np.where(h)[0]) >0:
        h_s,h_e = np.where(h)[0][[0,-1]]
    else:
        h_s = h_e = 0

    if len(np.where(w)[0]) >0:
        w_s,w_e = np.where(w)[0][[0,-1]]
    else:
        w_s = w_e = 0
    return d_s, d_e, h_s, h_e, w_s, w_e


P = osp.expanduser("~/data/verse")
SAVE_P = osp.join(P, "processed-verse19")
SAVE_NII_P = osp.join(SAVE_P, "nii")
SAVE_NPY_P = osp.join(SAVE_P, "npy")

for subset in ("test", "training", "validation"):
    print('\t', subset)
    subset_d = f"dataset-verse19{subset}"
    image_d = osp.join(P, subset_d, "rawdata")
    label_d = osp.join(P, subset_d, "derivatives")

    save_nii_p = osp.join(SAVE_NII_P, subset)
    save_npy_p = osp.join(SAVE_NPY_P, subset)
    os.makedirs(save_nii_p, exist_ok=True)
    os.makedirs(save_npy_p, exist_ok=True)

    for vol_name in os.listdir(image_d):
        if not vol_name.startswith("sub-verse"):
            continue
        print(vol_name)
        for image_path in glob.glob(osp.join(image_d, vol_name, "*.nii.gz")):
            bn = osp.basename(image_path)
            print(bn, end='\r')
            # sub-verse414_split-verse273_ct.nii.gz
            fid = bn[:-10]
            # sub-verse414_split-verse273_seg-vert_msk.nii.gz
            label_path = osp.join(label_d, vol_name, f"{fid}_seg-vert_msk.nii.gz")
            assert osp.isfile(image_path), f"* No label file: {label_path} of image {image_path}"

            image_arr = read_reorient2RAI(image_path)
            label_arr = read_reorient2RAI(label_path)

            image_arr = image_arr.astype(np.float32)
            # label_arr = convert_labels(label_arr)

            d_s, d_e, h_s, h_e, w_s, w_e = getRangeImageDepth(label_arr)
            d, h, w = image_arr.shape

            d_s = (d_s - 4).clip(min=0, max=d)
            d_e = (d_e + 4).clip(min=0, max=d)
            h_s = (h_s - 4).clip(min=0, max=h)
            h_e = (h_e + 4).clip(min=0, max=h)
            w_s = (w_s - 4).clip(min=0, max=w)
            w_e = (w_e + 4).clip(min=0, max=w)

            image_arr = image_arr[d_s:d_e, h_s:h_e, w_s: w_e]
            label_arr = label_arr[d_s:d_e, h_s:h_e, w_s: w_e]

            upper_bound_intensity_level = np.percentile(image_arr, 98)

            image_arr = image_arr.clip(min=0, max=upper_bound_intensity_level)
            image_arr = (image_arr - image_arr.mean()) / (image_arr.std() + 1e-8)

            # dn, hn, wn = image_arr.shape
            # image_arr = zoom(image_arr, [144/dn, 144/hn, 144/wn], order=0)
            # label_arr = zoom(label_arr, [144/dn, 144/hn, 144/wn], order=0)

            image = sitk.GetImageFromArray(image_arr)
            label = sitk.GetImageFromArray(label_arr)

            # save .nii.gz
            sitk.WriteImage(image, os.path.join(save_nii_p, osp.basename(image_path)))
            sitk.WriteImage(label, os.path.join(save_nii_p, osp.basename(label_path)))
            # save .npy
            np.save(os.path.join(save_npy_p, osp.basename(image_path)[:-7]), image_arr)
            np.save(os.path.join(save_npy_p, osp.basename(label_path)[:-7]), label_arr)
