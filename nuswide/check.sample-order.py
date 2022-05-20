import os.path as osp
import re


P = "G:/dataset/NUSWIDE"
IMAGE_LIST_F = osp.join(P, "ImageList", "Imagelist.txt")
TAG_LIST_F = osp.join(P, "NUS_WID_Tags", "All_Tags.txt")

img_id_pat = re.compile(r"[\_\-\w]+\\[0-9]+\_([0-9]+)\.jpg\s+")
n = 0
with open(IMAGE_LIST_F, "r") as img_f, \
        open(TAG_LIST_F, "r", encoding='utf-8') as txt_f:
    for i, (ln_img, ln_txt) in enumerate(zip(img_f, txt_f)):
        # print(ln_img, '\n', ln_txt)
        m_obj = img_id_pat.match(ln_img)
        assert m_obj is not None
        img_id = int(m_obj.group(1))
        txt_id = int(ln_txt.split()[0])
        # print(img_id, txt_id)
        n += 1
        if img_id != txt_id:
            print("* DIFF: #{}: {}, {}".format(i, img_id, txt_id))
print("DONE:", n)
