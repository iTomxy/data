import os
import os.path as osp
import platform


"""
soft-link all images into a single folder,
with the name modified to their (0-base) sample ID.

Hint:
Use RELATIVE source path in linux,
then you can simply soft-link that `images/` in any project
instead of creating a new project-specific `images/`.
But this trick does NOT work in Windows.
"""


# convert path seperator
cvt_sep = lambda p: p.replace('\\/'.replace(os.sep, ''), os.sep)


P = cvt_sep("/usr/local/dataset/COCO")
SPLIT = ["val2017", "train2017"]
IMAGE_DEST = osp.join(P, "images")  # path you place `images/` in
if not osp.exists(IMAGE_DEST):
    os.makedirs(IMAGE_DEST)


id_map_data = {}
with open(osp.join(P, "id-map.COCO.txt"), "r") as f:
    for _new_id, line in enumerate(f):
        _old_id, *_ = line.strip().split()
        id_map_data[int(_old_id)] = _new_id
N_DATA = len(id_map_data)
print("#data:", N_DATA)  # 123,287


_cnt = 0
for split in SPLIT:
    IMAGE_SRC_ABS = osp.join(P, split)  # absolute source path
    IMAGE_SRC_REL = osp.join("..", split)  # relative source path

    print("soft-linking:", IMAGE_SRC_ABS, "->", IMAGE_DEST)
    for f in os.listdir(IMAGE_SRC_ABS):
        old_id = int(f.split(".jpg")[0])
        new_id = id_map_data[old_id]
        img_p = osp.join(IMAGE_SRC_REL, f)
        new_img_p = osp.join(IMAGE_DEST, "{}.jpg".format(new_id))
        os.symlink(img_p, new_img_p)
        _cnt += 1
        if _cnt % 1000 == 0:
            print(_cnt)
print("DONE")
