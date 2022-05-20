import os
import os.path as osp
import numpy as np
import scipy.io as sio


P = "G:/dataset/NUSWIDE"

id_cls = {}
with open(osp.join(P, "Concepts81.txt"), "r") as f:
    for cid, line in enumerate(f):
        cn = line.strip()
        id_cls[cid] = cn
# print("\nclass-ID:", cls_id)
labels = sio.loadmat(osp.join(P, "labels.nuswide.mat"))["labels"]

lab_sum = labels.sum(0)
# print("label sum:", lab_sum)
class_desc = np.argsort(lab_sum)[::-1]
tc21 = np.sort(class_desc[:21])
tc10 = np.sort(class_desc[:10])
print("TC-21:", {id_cls[k]: lab_sum[k] for k in tc21})
print("TC-10:", {id_cls[k]: lab_sum[k] for k in tc10})


def make_sub_class(tc):
    n_top = len(tc)
    print("-> process TC-{}".format(n_top))
    with open(osp.join(P, "class-name.nuswide-tc{}.txt".format(n_top)), "w") as f:
        for i in range(n_top):
            cid = tc[i]
            cn = id_cls[cid]
            n_sample = lab_sum[cid]
            # format: <new class id> <class name> <original class id> <#sample>
            f.write("{} {} {} {}\n".format(i, cn, cid, n_sample))

    sub_labels = labels[:, tc]
    print("sub labels:", sub_labels.shape, ", cardinality:", sub_labels.sum())
    # sub labels: (269648, 21) , cardinality: 411438
    # sub labels: (269648, 10) , cardinality: 332189
    sio.savemat(osp.join(P, "labels.nuswide-tc{}.mat".format(n_top)),
        {"labels": sub_labels}, do_compression=True)

make_sub_class(tc21)
make_sub_class(tc10)
