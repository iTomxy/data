import os
import os.path as osp
import numpy as np
import scipy.io as sio


P = "G:/dataset/NUSWIDE"
labels_10 = sio.loadmat(osp.join(P, "labels.nuswide-tc10.mat"))["labels"]
labels_21 = sio.loadmat(osp.join(P, "labels.nuswide-tc21.mat"))["labels"]
texts_1 = sio.loadmat(osp.join(P, "texts.nuswide.All_Tags.mat"))["texts"]
texts_2 = sio.loadmat(osp.join(P, "texts.nuswide.AllTags1k.mat"))["texts"]


def pick_clean(label, text, name, double_sieve):
    clean_id = []
    on_map = {}
    new_id = 0
    for i, (l, t) in enumerate(zip(label, text)):
        # if only sieved by label (`double_sieve` = False)
        # we get 195,834 samples in TC-21, and 186,577 in TC-10
        # which matches the one DCMH provided
        if (0 == l.sum()):
            continue
        # if sieved by both label & text (`double_sieve` = True)
        # we got 190,421 samples in TC-21, and 181,365 in TC-10
        if double_sieve and (0 == t.sum()):
            continue
        on_map[new_id] = i
        new_id += 1
        clean_id.append(i)
    clean_id = np.asarray(clean_id).astype(np.int32)
    print(name, clean_id.shape)
    sio.savemat(osp.join(P, "clean_id.nuswide.{}.mat".format(name)),
        {"clean_id": clean_id}, do_compression=True)

    # with open(osp.join(P, "clean-full-map.nuswide.{}.txt".format(name)), "w") as f:
    #     for k in on_map:
    #         # format: <clean id> <full id>
    #         f.write("{} {}\n".format(k, on_map[k]))


for label, ln in zip([labels_21, labels_10], ["tc21", "tc10"]):
    # single sieving (label only)
    pick_clean(label, label, ln, False)
    # double sieving (label + text)
    for text, tn in zip([texts_1, texts_2], ["All_Tags", "AllTags1k"]):
        pick_clean(label, text, "{}.{}".format(ln, tn), True)
