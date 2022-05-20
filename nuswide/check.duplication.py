import os
import os.path as osp
import numpy as np
import scipy.io as sio


P = "G:/dataset/NUSWIDE"
IMAGE_LIST = osp.join(P, "ImageList", "Imagelist.txt")


print("find duplications")
img_id = {}  # image path -> serial sample id (increasing)
# path format: `actor\0001_2124494179.jpg`
# in which `2124494179` is treated as image id
id_img = {}  # image id -> image path
# record duplicated ID pairs, and the corresponding image ID
idx_a, idx_b, id_im = [], [], []
with open(IMAGE_LIST, "r") as f:
    for sid, line in enumerate(f):
        line = line.strip()
        if line:
            img_id[line] = sid

            img_f = line.split("\\")[1].split("_")[1]
            _id = int(img_f.split(".")[0])
            if _id in id_img:
                print("duplicated:", _id, line, id_img[_id])
                _idx_a, _idx_b = img_id[id_img[_id]], sid
                idx_a.append(_idx_a)
                idx_b.append(_idx_b)
                #print("sid pair:", _idx_a, _idx_b)
                id_im.append(_id)
            else:
                id_img[_id] = line

print("unique id:", len(set(id_img.keys())))  # 269642
idx_a = np.asarray(idx_a)
idx_b = np.asarray(idx_b)
print("duplicated index pairs:", idx_a, idx_b)


print("check the consistency between duplicated pairs")
id_cls = {}
with open(osp.join(P, "Concepts81.txt"), "r") as f:
    for cid, line in enumerate(f):
        cn = line.strip()
        id_cls[cid] = cn
labels = sio.loadmat(osp.join(P, "labels.nuswide.mat"))["labels"]
texts = sio.loadmat(osp.join(P, "texts.nuswide.AllTags1k.mat"))["texts"]
for _idx_a, _idx_b, _id_im in zip(idx_a, idx_b, id_im):
    la, lb = labels[_idx_a], labels[_idx_b]
    label_diff = (la != lb).sum()
    if 0 != label_diff:
        print("\tlabel diff:", _id_im, _idx_a, _idx_b, label_diff)
        print("class set 1:", [id_cls[c] for c in range(la.shape[0]) if (la[c] > 0)])
        print("class set 2:", [id_cls[c] for c in range(lb.shape[0]) if (lb[c] > 0)])

    text_diff = (texts[_idx_a] != texts[_idx_b]).sum()
    if 0 != text_diff:
        print("text diff:", _id_im, _idx_a, _idx_b, text_diff)
