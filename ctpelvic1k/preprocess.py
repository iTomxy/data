import os, os.path as osp, pprint, json, argparse, sys, glob
from collections import defaultdict
import numpy as np
import SimpleITK as sitk
import itk

sys.path.append("../verse")
from complete_label import predict

"""CTPelvic1K
Unify orientation, predict with pretrained TotalSegmentator model for unlabelled bones, and slice along the IS dimension.
From: https://github.com/xmed-lab/GenericSSL/blob/main/code/data/preprocess_mmwhs.py
"""

def read_reorient2LPS(path):
    """reorient using itk"""
    itk_img = itk.imread(path)

    filter = itk.OrientImageFilter.New(itk_img)
    filter.UseImageDirectionOn()
    filter.SetInput(itk_img)
    m = itk.Matrix[itk.D, 3, 3]()
    m.SetIdentity() # identity direction matrix gives LPS
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


def proc_volume(SAVE_NII_P, image_path, label_path, save_fid):
    """process & save 1 volume"""
    if  osp.isfile(os.path.join(SAVE_NII_P, f"{save_fid}_image.nii.gz")) and \
        osp.isfile(os.path.join(SAVE_NII_P, f"{save_fid}_label.nii.gz")): #and \
        # osp.isfile(os.path.join(SAVE_NPY_P, f"{save_fid}_image.npy")) and \
        # osp.isfile(os.path.join(SAVE_NPY_P, f"{save_fid}_label.npy")):
        return

    image_itk, image_arr = read_reorient2LPS(image_path)
    label_itk, label_arr = read_reorient2LPS(label_path)

    image_arr = image_arr.astype(np.float32)
    # label_arr = convert_labels(label_arr)
    label_arr = label_arr.astype(np.uint8) # class id in [0, 4]

    d_s, d_e, h_s, h_e, w_s, w_e = getRangeImageDepth(label_arr)
    d, h, w = image_arr.shape

    # d_s = (d_s - 4).clip(min=0, max=d)
    # d_e = (d_e + 4).clip(min=0, max=d)
    d_s = np.clip(d_s, 0, d)
    d_e = np.clip(d_e, 0, d)
    h_s = (h_s - 4).clip(min=0, max=h)
    h_e = (h_e + 4).clip(min=0, max=h)
    w_s = (w_s - 4).clip(min=0, max=w)
    w_e = (w_e + 4).clip(min=0, max=w)

    image_arr = image_arr[d_s:d_e, h_s:h_e, w_s: w_e]
    label_arr = label_arr[d_s:d_e, h_s:h_e, w_s: w_e]

    upper_bound_intensity_level = np.percentile(image_arr, 98)
    image_arr = image_arr.clip(min=0, max=upper_bound_intensity_level)

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


def reorient_data(args):
    """reorient data & save in a single folder"""
    err_log = defaultdict(list)
    SAVE_NII_P = os.path.join(args.data_path, "reorient_LPS")
    os.makedirs(SAVE_NII_P, exist_ok=True)

    print("1 abdomen")
    data_p = osp.join(args.data_path, "dataset1_abdomen/RawData")
    label_p = osp.join(args.data_path, "dataset1_mask_mappingback")
    cnt = 0
    for subset in os.listdir(data_p):
        print('\t', subset)
        subset_p = osp.join(data_p, subset, "img")
        for f in os.listdir(subset_p):
            # img0001.nii.gz, dataset1_img0001_mask_4label.nii.gz
            fid = f[:-len(".nii.gz")]
            lab_f = osp.join(label_p, f"dataset1_{fid}_mask_4label.nii.gz")
            new_fid = fid[len("img"):]
            if osp.isfile(lab_f):
                print(f, end='\r')
                proc_volume(SAVE_NII_P, osp.join(subset_p, f), lab_f, f"d1_{new_fid}")
                cnt += 1
    print("d1:", cnt)


    print("2 colonog (NOT DOWNLOAD YET)")


    print("3 MSD-T10")
    data_p = osp.join(args.data_path, "dataset3_msd-t10")
    label_p = osp.join(args.data_path, "dataset3_mask_mappingback")
    cnt = 0
    for subset in ("imagesTr", "imagesTs"):
        print('\t', subset)
        subset_p = osp.join(data_p, subset)
        for f in os.listdir(subset_p):
            # colon_001.nii.gz, dataset3_colon_001_mask_4label.nii.gz
            fid = f[:-len(".nii.gz")]
            lab_f = osp.join(label_p, f"dataset3_{fid}_mask_4label.nii.gz")
            new_fid = fid[len("colon_"):]
            if osp.isfile(lab_f):
                print(f, end='\r')
                proc_volume(SAVE_NII_P, osp.join(subset_p, f), lab_f, f"d3_{new_fid}")
                cnt += 1
    print("d3:", cnt)


    print("4 kits19")
    data_p = osp.join(args.data_path, "dataset4_kits19")
    label_p = osp.join(args.data_path, "dataset4_mask_mappingback")
    cnt = 0
    for fid in os.listdir(data_p):
        # case_00014/imaging.nii.gz, dataset4_case_00014_mask_4label.nii.gz
        img_f = osp.join(data_p, fid, "imaging.nii.gz")
        assert osp.isfile(img_f), img_f
        lab_f = osp.join(label_p, f"dataset4_{fid}_mask_4label.nii.gz")
        if osp.isfile(lab_f):
            print(fid, end='\r')
            new_fid = fid[len("case_"):]
            proc_volume(SAVE_NII_P, img_f, lab_f, f"d4_{new_fid}")
            cnt += 1
    print("d4:", cnt)


    print("5 cervix")
    data_p = osp.join(args.data_path, "dataset5_cervix/RawData")
    label_p = osp.join(args.data_path, "dataset5_mask_mappingback")
    cnt = 0
    for subset in os.listdir(data_p):
        print('\t', subset)
        subset_p = osp.join(data_p, subset, "img")
        for f in os.listdir(subset_p):
            # 0507688-Image.nii.gz, dataset5_0507688_Image_mask_4label.nii.gz
            fid = f[:-len(".nii.gz")].replace('-', '_')
            lab_f = osp.join(label_p, f"dataset5_{fid}_mask_4label.nii.gz")
            if osp.isfile(lab_f):
                print(f, end='\r')
                new_fid = fid[:-len("_Image")]
                proc_volume(SAVE_NII_P, osp.join(subset_p, f), lab_f, f"d5_{new_fid}")
                cnt += 1
    print("d5:", cnt)


    print("6 clinic")
    data_p = osp.join(args.data_path, "dataset6_clinic")
    label_p = osp.join(args.data_path, "dataset6_mask_mappingback")
    cnt = 0
    for f in os.listdir(data_p):
        # dataset6_CLINIC_0001_data.nii.gz, dataset6_CLINIC_0001_mask_4label.nii.gz
        fid = f[:-len("_data.nii.gz")]
        lab_f = osp.join(label_p, f"{fid}_mask_4label.nii.gz")
        if osp.isfile(lab_f):
            print(f, end='\r')
            try:
                new_fid = fid[len("dataset6_CLINIC_"):]
                proc_volume(SAVE_NII_P, osp.join(data_p, f), lab_f, f"d6_{new_fid}")
            except itk.support.extras.TemplateTypeError:
                # TemplateTypeError: itk.OrientImageFilter is not wrapped for input type `None`.
                err_log["itk.support.extras.TemplateTypeError"].append(osp.join(data_p, f))
            cnt += 1
    print("d6:", cnt)


    print("7 clinic metal")
    data_p = osp.join(args.data_path, "dataset7_clinic_metal")
    label_p = osp.join(args.data_path, "dataset7_mask_mappingback")
    cnt = 0
    for f in os.listdir(data_p):
        # dataset7_CLINIC_metal_0000_data.nii.gz, CLINIC_metal_0000_mask_4label.nii.gz
        fid = f[len("dataset7_"): -len("_data.nii.gz")]
        lab_f = osp.join(label_p, f"{fid}_mask_4label.nii.gz")
        if osp.isfile(lab_f):
            print(f, end='\r')
            try:
                new_fid = fid[len("CLINIC_metal_")]
                proc_volume(SAVE_NII_P, osp.join(data_p, f), lab_f, f"d7_{new_fid}")
            except itk.support.extras.TemplateTypeError:
                # TemplateTypeError: itk.OrientImageFilter is not wrapped for input type `None`.
                err_log["itk.support.extras.TemplateTypeError"].append(osp.join(data_p, f))
            cnt += 1
    print("d7:", cnt)


    print("\terror log")
    pprint.pprint(err_log)
    with open("error-log.json", "w") as f:
        json.dump(err_log, f, indent=2)


def ts_infer(args):
    """segment with pretrained TotalSegmentator for missing bones (e.g. spine)"""
    src_path = os.path.join(args.data_path, "reorient_LPS")
    assert os.path.isdir(src_path), "Path to reoriented data does not exist: {}".format(src_path)

    save_path = os.path.join(args.data_path, "ts_pred")
    os.makedirs(save_path, exist_ok=True)
    for f in glob.iglob(os.path.join(src_path, "*_image.nii.gz")):
        vid = os.path.basename(f)[: -len("_image.nii.gz")]
        predict(f, vid, save_path, subtasks=args.ts_tasks)
        print(vid, end='\r')


def slice_data(args):
    """slice along the IS axis"""
    src_path = os.path.join(args.data_path, "reorient_LPS")
    assert os.path.isdir(src_path), "Path to reoriented data does not exist: {}".format(src_path)

    ts_pred_path = os.path.join(args.data_path, "ts_pred")
    assert os.path.isdir(ts_pred_path), "TotalSegmentator predictions do not exist: {}".format(ts_pred_path)

    save_path = os.path.join(args.data_path, "slice_is")

    for f in glob.iglob(os.path.join(src_path, "*_image.nii.gz")):
        vid = os.path.basename(f)[: -len("_image.nii.gz")]
        save_dir = os.path.join(save_path, vid)
        if os.path.isdir(save_dir):
            continue

        image = sitk.ReadImage(f)
        label = sitk.ReadImage(os.path.join(src_path, "{}_label.nii.gz").format(vid))
        # axis order changes from LPS to SLP (or SPL?) when converting to numpy
        ts_preds = {
            t: sitk.ReadImage(os.path.join(ts_pred_path, "{}-{}.nii.gz".format(vid, t)))
            for t in args.ts_tasks
        }

        image_np = sitk.GetArrayFromImage(image)
        label_np = sitk.GetArrayFromImage(label)
        ts_preds = {k: sitk.GetArrayFromImage(v) for k, v in ts_preds.items()}
        assert image_np.shape == label_np.shape
        for k, v in ts_preds.items():
            assert image_np.shape == v.shape, "Shape mismatch: image ({}) vs. {} ({})".format(
                image_np.shape, k, v.shape)
        # print(image.shape)

        tmp_dir = save_dir + ".tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        for i in range(image_np.shape[0]):
            _dict = dict(
                spacing=image.GetSpacing(),
                image=image_np[i],
                label=label_np[i]
            )
            for k, v in ts_preds.items():
                _dict[k] = v[i]
            np.savez_compressed(
                os.path.join(tmp_dir, str(i)),
                **_dict
            )

        os.rename(tmp_dir, save_dir)
        print(vid, end='\r')


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str, default="~/sd10t/ctpelvic1k")
    parser.add_argument('-t', '--ts-tasks', type=str, nargs='+', default=["total"],
        choices=["total", "appendicular_bones", "vertebrae_body"],
        help="do what types of TotalSegmentator prediction.")
    args = parser.parse_args()

    args.data_path = osp.expanduser(args.data_path)
    assert os.path.isdir(args.data_path), args.data_path

    reorient_data(args)
    ts_infer(args)
    slice_data(args)
