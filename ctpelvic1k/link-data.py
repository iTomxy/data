import os, os.path as osp, glob
import shutil


"""spliting from:
- (CARS 2021) Deep learning to segment pelvic bones: large-scale CT datasets and baseline models
- https://github.com/MIRACLE-Center/CTPelvic1K
"""


P = osp.expanduser("~/data/CTPelvic1K")
LINK_P = osp.expanduser("~/data/project-data/ctpelvic1k")
SUBSETS = ("train", "val", "test")
for subset in SUBSETS:
    os.makedirs(osp.join(LINK_P, subset), exist_ok=True)


"""dataset 1 Abdomen"""
data_p = osp.join(P, "dataset1_abdomen/RawData")
mask_p = osp.join(P, "dataset1_mask_mappingback")
n_train, n_val, n_test = 21, 7, 7

label_files = os.listdir(mask_p)
# dataset1_img0001_mask_4label.nii.gz
label_files = sorted(label_files, key=lambda _f: int(_f.split('_')[1][3:]))
chosen_ids = [f.split("_")[1] for f in label_files]
src_data_files = []
for subset in ("Training", "Testing"):
    subset_p = osp.join(data_p, subset, "img")
    for f in os.listdir(subset_p):
        # img0001.nii.gz
        img_id = f.split('.')[0]
        if img_id in chosen_ids:
            src_data_files.append(osp.join(subset_p, f))

# soft-link
assert len(src_data_files) == n_train + n_test + n_val
src_data_files = sorted(src_data_files, key=lambda _f: int(osp.basename(_f).split('.')[0][3:]))
for i, (data_f, mask_f) in enumerate(zip(src_data_files, label_files)):
    subset = SUBSETS[int(i >= n_train) + int(i >= n_train + n_val)]
    img_id = osp.basename(data_f).split('.')[0]
    os.symlink(data_f, osp.join(LINK_P, subset, f"dataset1_{img_id}_data.nii.gz"))
    os.symlink(osp.join(mask_p, mask_f), osp.join(LINK_P, subset, mask_f))


"""dataset 2 Colonog"""
print("(2023.9.18) image not downloaded yet")


"""dataset 3 MSD-T10"""
data_p = osp.join(P, "dataset3_msd-t10")
mask_p = osp.join(P, "dataset3_mask_mappingback")
n_train, n_val, n_test = 93, 31, 31

label_files = os.listdir(mask_p)
# dataset3_colon_001_mask_4label.nii.gz
label_files = sorted(label_files, key=lambda _f: int(_f.split('_')[2]))
chosen_ids = ['_'.join(f.split("_")[1:3]) for f in label_files]
src_data_files = []
for subset in ("imagesTr", "imagesTs"):
    subset_p = osp.join(data_p, subset)
    for f in os.listdir(subset_p):
        # colon_001.nii.gz
        img_id = f.split('.')[0]
        if img_id in chosen_ids:
            src_data_files.append(osp.join(subset_p, f))

# soft-link
assert len(src_data_files) == n_train + n_test + n_val
src_data_files = sorted(src_data_files, key=lambda _f: int(osp.basename(_f).split('.')[0].split('_')[1]))
for i, (data_f, mask_f) in enumerate(zip(src_data_files, label_files)):
    subset = SUBSETS[int(i >= n_train) + int(i >= n_train + n_val)]
    img_id = osp.basename(data_f).split('.')[0]
    os.symlink(data_f, osp.join(LINK_P, subset, f"dataset3_{img_id}_data.nii.gz"))
    os.symlink(osp.join(mask_p, mask_f), osp.join(LINK_P, subset, mask_f))


"""dataset 4 KiTS'19"""
data_p = osp.join(P, "dataset4_kits19")
mask_p = osp.join(P, "dataset4_mask_mappingback")
n_train, n_val, n_test = 26, 9, 9

label_files = os.listdir(mask_p)
# dataset4_case_00014_mask_4label.nii.gz
label_files = sorted(label_files, key=lambda _f: int(_f.split('_')[2]))
chosen_ids = ['_'.join(f.split("_")[1:3]) for f in label_files]
src_data_files = []
for img_id in os.listdir(data_p):
    # case_00014/imaging.nii.gz
    if img_id in chosen_ids:
        src_data_files.append(osp.join(data_p, img_id, "imaging.nii.gz"))

# soft-link
assert len(src_data_files) == n_train + n_test + n_val
src_data_files = sorted(src_data_files, key=lambda _f: int(osp.basename(osp.dirname(_f)).split('_')[1]))
for i, (data_f, mask_f) in enumerate(zip(src_data_files, label_files)):
    subset = SUBSETS[int(i >= n_train) + int(i >= n_train + n_val)]
    img_id = osp.basename(osp.dirname(data_f))
    os.symlink(data_f, osp.join(LINK_P, subset, f"dataset4_{img_id}_data.nii.gz"))
    os.symlink(osp.join(mask_p, mask_f), osp.join(LINK_P, subset, mask_f))


"""dataset 5 Cervix"""
data_p = osp.join(P, "dataset5_cervix/RawData")
mask_p = osp.join(P, "dataset5_mask_mappingback")
n_train, n_val, n_test = 24, 8, 9

label_files = os.listdir(mask_p)
# dataset5_0507688_Image_mask_4label.nii.gz
label_files = sorted(label_files, key=lambda _f: int(_f.split('_')[1]))
chosen_ids = ['_'.join(f.split("_")[1:3]) for f in label_files]
src_data_files = []
for subset in ("Training", "Testing"):
    subset_p = osp.join(data_p, subset, "img")
    for f in os.listdir(subset_p):
        # 0507688-Image.nii.gz
        img_id = f.split('.')[0].replace('-', '_')
        if img_id in chosen_ids:
            src_data_files.append(osp.join(subset_p, f))

# soft-link
assert len(src_data_files) == n_train + n_test + n_val
src_data_files = sorted(src_data_files, key=lambda _f: int(osp.basename(_f).split('-')[0]))
for i, (data_f, mask_f) in enumerate(zip(src_data_files, label_files)):
    subset = SUBSETS[int(i >= n_train) + int(i >= n_train + n_val)]
    img_id = osp.basename(data_f).split('.')[0].replace('-', '_')
    os.symlink(data_f, osp.join(LINK_P, subset, f"dataset5_{img_id}_data.nii.gz"))
    os.symlink(osp.join(mask_p, mask_f), osp.join(LINK_P, subset, mask_f))


"""dataset 6 Clinic"""
data_p = osp.join(P, "dataset6_clinic")
mask_p = osp.join(P, "dataset6_mask_mappingback")
n_train, n_val, n_test = 61, 21, 21

label_files = os.listdir(mask_p)
# dataset6_CLINIC_0001_mask_4label.nii.gz
label_files = sorted(label_files, key=lambda _f: int(_f.split('_')[2]))
chosen_ids = ['_'.join(f.split("_")[1:3]) for f in label_files]
for i, (data_f, mask_f) in enumerate(zip(
    # dataset6_CLINIC_0001_data.nii.gz
    sorted(os.listdir(data_p), key=lambda _f: int(_f.split('_')[2])),
    label_files
)):
    img_id = '_'.join(data_f.split("_")[1:3])
    assert img_id in chosen_ids, img_id
    # soft-link
    subset = SUBSETS[int(i >= n_train) + int(i >= n_train + n_val)]
    os.symlink(osp.join(data_p, data_f), osp.join(LINK_P, subset, data_f))
    os.symlink(osp.join(mask_p, mask_f), osp.join(LINK_P, subset, mask_f))


"""dataset 7 Clinic-metal"""
data_p = osp.join(P, "dataset7_clinic_metal")
mask_p = osp.join(P, "dataset7_mask_mappingback")
n_train, n_val, n_test = 0, 0, 14

label_files = os.listdir(mask_p)
# CLINIC_metal_0000_mask_4label.nii.gz
label_files = sorted(label_files, key=lambda _f: int(_f.split('_')[2]))
chosen_ids = ['_'.join(f.split("_")[:3]) for f in label_files]
for i, (data_f, mask_f) in enumerate(zip(
    # dataset7_CLINIC_metal_0000_data.nii.gz
    sorted(os.listdir(data_p), key=lambda _f: int(_f.split('_')[3])),
    label_files
)):
    img_id = '_'.join(data_f.split("_")[1:4])
    if img_id in chosen_ids:
        # soft-link
        subset = SUBSETS[int(i >= n_train) + int(i >= n_train + n_val)]
        os.symlink(osp.join(data_p, data_f), osp.join(LINK_P, subset, data_f))
        os.symlink(osp.join(mask_p, mask_f), osp.join(LINK_P, subset, mask_f))
