import os
import os.path as osp
import numpy as np
import scipy.io as sio


N_SAMPLE = 269648
P = "G:/dataset/NUSWIDE"
LABEL_P = osp.join(P, "Groundtruth/AllLabels")


# class-ID correspondance
# class order determined by `Concepts81.txt`
cls_id = {}
with open(osp.join(P, "Concepts81.txt"), "r") as f:
    for cid, line in enumerate(f):
        cn = line.strip()
        cls_id[cn] = cid
# print("\nclass-ID:", cls_id)
id_cls = {cls_id[k]: k for k in cls_id}
# print("\nID-class:", id_cls)
N_CLASS = len(cls_id)
print("\n#classes:", N_CLASS)

class_files = os.listdir(LABEL_P)
# print("\nlabel file:", len(class_files), class_files)
label_key = lambda x: x.split(".txt")[0].split("Labels_")[-1]

labels = np.zeros([N_SAMPLE, N_CLASS], dtype=np.uint8)
for cf in class_files:
    c_name = label_key(cf)
    cid = cls_id[c_name]
    print('->', cid, c_name)
    with open(osp.join(LABEL_P, cf), "r") as f:
        for sid, line in enumerate(f):
            if int(line) > 0:
                labels[sid][cid] = 1
print("labels:", labels.shape, ", cardinality:", labels.sum())
# labels: (269648, 81) , cardinality: 503848
sio.savemat(osp.join(P, "labels.nuswide.mat"), {"labels": labels}, do_compression=True)
