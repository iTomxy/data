import os, os.path as osp, glob, pprint, json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
# import SimpleITK as sitk
import itk


"""
From: https://github.com/xmed-lab/GenericSSL/blob/main/code/data/preprocess_mmwhs.py
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


P = osp.expanduser("~/data/ctpelvic1k")
MODE = "none"
assert MODE in ("none", "window+std", "norm"), MODE
if "none" == MODE:
    SAVE_P = osp.join(P, "processed-ctpelvic1k")
elif "window+std" == MODE: # windowing + standardisation
    WINDOW_LEVEL = 300
    WINDOW_WIDTH = 290
    WINDOW_MIN = WINDOW_LEVEL - WINDOW_WIDTH
    WINDOW_MAX = WINDOW_LEVEL + WINDOW_WIDTH
    SAVE_P = osp.join(P, f"processed-ctpelvic1k-wl{WINDOW_LEVEL}-ww{WINDOW_WIDTH}-std")
elif "norm" == MODE: # normalisation
    SAVE_P = osp.join(P, "processed-ctpelvic1k-norm")
SAVE_NII_P = SAVE_P # osp.join(SAVE_P, "nii")
# SAVE_NPY_P = osp.join(SAVE_P, "npy")
os.makedirs(SAVE_NII_P, exist_ok=True)
# os.makedirs(SAVE_NPY_P, exist_ok=True)


def proc_volume(image_path, label_path, save_fid):
    """process & save 1 volume"""
    if  osp.isfile(os.path.join(SAVE_NII_P, f"{save_fid}_image.nii.gz")) and \
        osp.isfile(os.path.join(SAVE_NII_P, f"{save_fid}_label.nii.gz")): #and \
        # osp.isfile(os.path.join(SAVE_NPY_P, f"{save_fid}_image.npy")) and \
        # osp.isfile(os.path.join(SAVE_NPY_P, f"{save_fid}_label.npy")):
        return

    image_itk, image_arr = read_reorient2RAI(image_path)
    label_itk, label_arr = read_reorient2RAI(label_path)

    image_arr = image_arr.astype(np.float32)
    # label_arr = convert_labels(label_arr)
    label_arr = label_arr.astype(np.uint8) # class id in [0, 4]

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
    if "norm" == MODE:
        image_arr = (image_arr - image_arr.mean()) / (image_arr.std() + np.finfo(np.float32).eps)
    elif "window+std" == MODE:
        image_arr = np.clip(image_arr, WINDOW_MIN, WINDOW_MAX)
        image_arr = (image_arr - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN) # in [0, 1]
        image_arr = (image_arr * 255).astype(np.uint8) # in [0, 255]

    # dn, hn, wn = image_arr.shape
    # image_arr = zoom(image_arr, [144/dn, 144/hn, 144/wn], order=0)
    # label_arr = zoom(label_arr, [144/dn, 144/hn, 144/wn], order=0)

    # save .npy
    # np.save(os.path.join(SAVE_NPY_P, f"{save_fid}_image.npy"), image_arr)
    # np.save(os.path.join(SAVE_NPY_P, f"{save_fid}_label.npy"), label_arr)

    # save .nii.gz
    image = itk.GetImageViewFromArray(image_arr)
    label = itk.GetImageViewFromArray(label_arr)
    # restore meta info
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    label.SetDirection(label_itk.GetDirection())
    label.SetSpacing(label_itk.GetSpacing())
    itk.imwrite(image, os.path.join(SAVE_NII_P, f"{save_fid}_image.nii.gz"))
    itk.imwrite(label, os.path.join(SAVE_NII_P, f"{save_fid}_label.nii.gz"))


err_log = defaultdict(list)

print("1 abdomen")
data_p = osp.join(P, "dataset1_abdomen/RawData")
label_p = osp.join(P, "dataset1_mask_mappingback")
cnt = 0
for subset in os.listdir(data_p):
    print('\t', subset)
    subset_p = osp.join(data_p, subset, "img")
    for f in os.listdir(subset_p):
        # img0001.nii.gz, dataset1_img0001_mask_4label.nii.gz
        fid = f[:-7]
        lab_f = osp.join(label_p, f"dataset1_{fid}_mask_4label.nii.gz")
        if osp.isfile(lab_f):
            print(f, end='\r')
            proc_volume(osp.join(subset_p, f), lab_f, f"d1_{fid}")
            cnt += 1
print("d1:", cnt)


print("2 colonog (NOT DOWNLOAD YET)")


print("3 MSD-T10")
data_p = osp.join(P, "dataset3_msd-t10")
label_p = osp.join(P, "dataset3_mask_mappingback")
cnt = 0
for subset in ("imagesTr", "imagesTs"):
    print('\t', subset)
    subset_p = osp.join(data_p, subset)
    for f in os.listdir(subset_p):
        # colon_001.nii.gz, dataset3_colon_001_mask_4label.nii.gz
        fid = f[:-7]
        lab_f = osp.join(label_p, f"dataset3_{fid}_mask_4label.nii.gz")
        if osp.isfile(lab_f):
            print(f, end='\r')
            proc_volume(osp.join(subset_p, f), lab_f, f"d3_{fid}")
            cnt += 1
print("d3:", cnt)


print("4 kits19")
data_p = osp.join(P, "dataset4_kits19")
label_p = osp.join(P, "dataset4_mask_mappingback")
cnt = 0
for fid in os.listdir(data_p):
    # case_00014/imaging.nii.gz, dataset4_case_00014_mask_4label.nii.gz
    img_f = osp.join(data_p, fid, "imaging.nii.gz")
    assert osp.isfile(img_f), img_f
    lab_f = osp.join(label_p, f"dataset4_{fid}_mask_4label.nii.gz")
    if osp.isfile(lab_f):
        print(fid, end='\r')
        proc_volume(img_f, lab_f, f"d4_{fid}")
        cnt += 1
print("d4:", cnt)


print("5 cervix")
data_p = osp.join(P, "dataset5_cervix/RawData")
label_p = osp.join(P, "dataset5_mask_mappingback")
cnt = 0
for subset in os.listdir(data_p):
    print('\t', subset)
    subset_p = osp.join(data_p, subset, "img")
    for f in os.listdir(subset_p):
        # 0507688-Image.nii.gz, dataset5_0507688_Image_mask_4label.nii.gz
        fid = f[:-7].replace('-', '_')
        lab_f = osp.join(label_p, f"dataset5_{fid}_mask_4label.nii.gz")
        if osp.isfile(lab_f):
            print(f, end='\r')
            proc_volume(osp.join(subset_p, f), lab_f, f"d5_{fid}")
            cnt += 1
print("d5:", cnt)


print("6 clinic")
data_p = osp.join(P, "dataset6_clinic")
label_p = osp.join(P, "dataset6_mask_mappingback")
cnt = 0
for f in os.listdir(data_p):
    # dataset6_CLINIC_0001_data.nii.gz, dataset6_CLINIC_0001_mask_4label.nii.gz
    fid = f[:-12]
    lab_f = osp.join(label_p, f"{fid}_mask_4label.nii.gz")
    if osp.isfile(lab_f):
        print(f, end='\r')
        try:
            fid = fid[9:] # rm prefix `dataset6_`
            proc_volume(osp.join(data_p, f), lab_f, f"d6_{fid}")
        except itk.support.extras.TemplateTypeError:
            # TemplateTypeError: itk.OrientImageFilter is not wrapped for input type `None`.
            err_log["itk.support.extras.TemplateTypeError"].append(osp.join(data_p, f))
        cnt += 1
print("d6:", cnt)


print("7 clinic metal")
data_p = osp.join(P, "dataset7_clinic_metal")
label_p = osp.join(P, "dataset7_mask_mappingback")
cnt = 0
for f in os.listdir(data_p):
    # dataset7_CLINIC_metal_0000_data.nii.gz, CLINIC_metal_0000_mask_4label.nii.gz
    fid = f[9:-12]
    lab_f = osp.join(label_p, f"{fid}_mask_4label.nii.gz")
    if osp.isfile(lab_f):
        print(f, end='\r')
        try:
            proc_volume(osp.join(data_p, f), lab_f, f"d7_{fid}")
        except itk.support.extras.TemplateTypeError:
            # TemplateTypeError: itk.OrientImageFilter is not wrapped for input type `None`.
            err_log["itk.support.extras.TemplateTypeError"].append(osp.join(data_p, f))
        cnt += 1
print("d7:", cnt)


print("\terror log")
pprint.pprint(err_log)
with open("error-log.json", "w") as f:
    json.dump(err_log, f, indent=2)
