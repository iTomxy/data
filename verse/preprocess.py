import os, os.path as osp, glob
import numpy as np
from tqdm import tqdm
# import SimpleITK as sitk
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
    # print("original:", itk_img.GetDirection(), '\n', itk_img.GetOrigin(), '\n', itk_img.GetSpacing())

    filter = itk.OrientImageFilter.New(itk_img)
    filter.UseImageDirectionOn()
    filter.SetInput(itk_img)
    m = itk.Matrix[itk.D, 3, 3]()
    m.SetIdentity()
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    itk_img = filter.GetOutput()
    # print("after:", itk_img.GetDirection(), '\n', itk_img.GetOrigin(), '\n', itk_img.GetSpacing())

    itk_arr = itk.GetArrayViewFromImage(itk_img)
    return itk_img, itk_arr


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
MODE = "window+std"
assert MODE in ("window+std", "norm"), MODE
if "window+std" == MODE: # windowing + standardisation
    WINDOW_LEVEL = 300
    WINDOW_WIDTH = 290
    WINDOW_MIN = WINDOW_LEVEL - WINDOW_WIDTH
    WINDOW_MAX = WINDOW_LEVEL + WINDOW_WIDTH
    SAVE_P = osp.join(P, f"processed-verse19-wl{WINDOW_LEVEL}-ww{WINDOW_WIDTH}-std")
elif "norm" == MODE: # normalisation
    SAVE_P = osp.join(P, "processed-verse19-norm")
SAVE_NII_P = SAVE_P # osp.join(SAVE_P, "nii")
# SAVE_NPY_P = osp.join(SAVE_P, "npy")

for subset in ("test", "training", "validation"):
    print('\t', subset)
    subset_d = f"dataset-verse19{subset}"
    image_d = osp.join(P, subset_d, "rawdata")
    label_d = osp.join(P, subset_d, "derivatives")

    save_nii_p = osp.join(SAVE_NII_P, subset)
    # save_npy_p = osp.join(SAVE_NPY_P, subset)
    os.makedirs(save_nii_p, exist_ok=True)
    # os.makedirs(save_npy_p, exist_ok=True)

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

            image_itk, image_arr = read_reorient2RAI(image_path)
            label_itk, label_arr = read_reorient2RAI(label_path)

            image_arr = image_arr.astype(np.float32)
            # label_arr = convert_labels(label_arr)
            label_arr = label_arr.astype(np.uint8) # class id in [0, 28]

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

            if "norm" == MODE:
                upper_bound_intensity_level = np.percentile(image_arr, 98)
                image_arr = image_arr.clip(min=0, max=upper_bound_intensity_level)
                image_arr = (image_arr - image_arr.mean()) / (image_arr.std() + 1e-8)
            elif "window+std" == MODE:
                image_arr = np.clip(image_arr, WINDOW_MIN, WINDOW_MAX)
                image_arr = (image_arr - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN) # in [0, 1]
                image_arr = (image_arr * 255).astype(np.uint8) # in [0, 255]

            # dn, hn, wn = image_arr.shape
            # image_arr = zoom(image_arr, [144/dn, 144/hn, 144/wn], order=0)
            # label_arr = zoom(label_arr, [144/dn, 144/hn, 144/wn], order=0)

            # save .npy
            # np.save(os.path.join(save_npy_p, fid+"_image"), image_arr)
            # np.save(os.path.join(save_npy_p, fid+"_label"), label_arr)

            # save .nii.gz
            image = itk.GetImageViewFromArray(image_arr)
            label = itk.GetImageViewFromArray(label_arr)
            # restore meta info
            image.SetDirection(image_itk.GetDirection())
            image.SetSpacing(image_itk.GetSpacing())
            label.SetDirection(label_itk.GetDirection())
            label.SetSpacing(label_itk.GetSpacing())
            # print("on save (image):", image.GetDirection(), '\n', image.GetOrigin(), '\n', image.GetSpacing())
            # print("on save (label):", label.GetDirection(), '\n', label.GetOrigin(), '\n', label.GetSpacing())
            itk.imwrite(image, os.path.join(save_nii_p, fid+"_image.nii.gz"))
            itk.imwrite(label, os.path.join(save_nii_p, fid+"_label.nii.gz"))
