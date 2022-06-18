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


P = cvt_sep("G:/dataset/NUSWIDE")
IMAGE_LIST = osp.join(P, "ImageList", "Imagelist.txt")
IMAGE_SRC_ABS = osp.join(P, "Flickr")  # absolute source path
IMAGE_SRC_REL = osp.join("..", "Flickr")  # relative source path
IMAGE_DEST = osp.join(P, "images")  # path you place `images/` in
if not osp.exists(IMAGE_DEST):
    os.makedirs(IMAGE_DEST)

# soft-linking command
if "Windows" == platform.system():
    cmd = "mklink {1} {0} > nul"
    img_src_path_pre = IMAGE_SRC_ABS
else:
    assert "Linux" == platform.system()
    cmd = "ln -s {0} {1}"
    img_src_path_pre = IMAGE_SRC_REL

print("soft-linking:", IMAGE_SRC_ABS, "->", IMAGE_DEST)
with open(IMAGE_LIST, "r") as f:
    for sid, line in enumerate(f):
        line = cvt_sep(line).strip()
        # img_p = osp.join(IMAGE_SRC_ABS, line)  # use this in Windows
        # img_p = osp.join(IMAGE_SRC_REL, line)  # use this in linux
        img_p = osp.join(img_src_path_pre, line)
        new_img_p = osp.join(IMAGE_DEST, "{}.jpg".format(sid))
        os.system(cmd.format(img_p, new_img_p))
        if sid % 1000 == 0:
            print(sid)
print("DONE")
