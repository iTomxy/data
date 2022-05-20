import os
import os.path as osp
import platform


"""
soft-link all images into a single folder,
with the name modified to their (0-base) sample ID.
"""


# convert path seperator
cvt_sep = lambda p: p.replace('\\/'.replace(os.sep, ''), os.sep)


P = cvt_sep("G:/dataset/NUSWIDE")
IMAGE_LIST = osp.join(P, "ImageList", "Imagelist.txt")
IMAGE_SRC = osp.join(P, "Flickr")
IMAGE_DEST = osp.join(os.getcwd(), "images")  # path you place `images/` in
if not osp.exists(IMAGE_DEST):
    os.makedirs(IMAGE_DEST)

# soft-linking command
if "Windows" == platform.system():
    cmd = "mklink {1} {0} > nul"
else:
    assert "Linux" == platform.system()
    cmd = "ln -s {0} {1}"

print("soft-linking:", IMAGE_SRC, "->", IMAGE_DEST)
with open(IMAGE_LIST, "r") as f:
    for sid, line in enumerate(f):
        line = cvt_sep(line).strip()
        img_p = osp.join(IMAGE_SRC, line)
        new_img_p = osp.join(IMAGE_DEST, "{}.jpg".format(sid))
        os.system(cmd.format(img_p, new_img_p))
        if sid % 1000 == 0:
            print(sid)
print("DONE")
