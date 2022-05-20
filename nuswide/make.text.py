import os
import os.path as osp
import numpy as np
import scipy.io as sio


N_SAMPLE = 269648
P = "G:/dataset/NUSWIDE"
TEXT_P = osp.join(P, "NUS_WID_Tags")


print("1st: TagList1k.txt (order) + All_Tags.txt (assignment)")
tag_id = {}
with open(osp.join(TEXT_P, "TagList1k.txt"), "r", encoding='utf-8') as f:
    for tid, line in enumerate(f):
        tn = line.strip()
        tag_id[tn] = tid
id_tag = {tag_id[k]: k for k in tag_id}
# print("\ntag-ID:", len(tag_id), list(tag_id)[:10])
N_TAG = len(id_tag)
print("\n#tag:", N_TAG)  # 1000

texts_1 = np.zeros([N_SAMPLE, N_TAG], dtype=np.uint8)
with open(os.path.join(TEXT_P, "All_Tags.txt"), "r", encoding='utf-8') as f:
    for sid, line in enumerate(f):
        # format: <sample id> <tags...>
        _tags = line.split()[1:]
        # print(_tags)
        for _t in _tags:
            if _t in tag_id:  # 限制在那 1k 个 tags 里
                tid = tag_id[_t]
                texts_1[sid][tid] = 1
        if sid % 1000 == 0:
            print(sid)
print("1st texts:", texts_1.shape, ", cardinality:", texts_1.sum())
# 1st texts: (269648, 1000) , cardinality: 1559503
sio.savemat(osp.join(P, "texts.nuswide.All_Tags.mat"), {"texts": texts_1}, do_compression=True)


print("2nd: AllTags1k.txt")
texts_2 = np.zeros([N_SAMPLE, N_TAG], dtype=np.uint8)
with open(os.path.join(TEXT_P, "AllTags1k.txt"), "r") as f:
    for sid, line in enumerate(f):
        # format: 1000-D space-seperated multi-hot vector
        line = list(map(int, line.split()))
        assert len(line) == 1000
        texts_2[sid] = np.asarray(line).astype(np.uint8)
        if sid % 1000 == 0:
            print(sid)
print("2nd texts:", texts_2.shape, ", cardinality:", texts_2.sum())
# 2nd texts: (269648, 1000) , cardinality: 1559464
sio.savemat(osp.join(P, "texts.nuswide.AllTags1k.mat"), {"texts": texts_2}, do_compression=True)


print("compare two methods")
# texts_1 = sio.loadmat(osp.join(P, "texts.nuswide.All_Tags.mat"))["texts"]
# texts_2 = sio.loadmat(osp.join(P, "texts.nuswide.AllTags1k.mat"))["texts"]
n_diff_order, n_diff_card = 0, 0
with open(osp.join(TEXT_P, "All_Tags.txt"), "r", encoding='utf-8') as f1, \
        open(osp.join(TEXT_P, "AllTags1k.txt"), "r") as f2:
    for i in range(texts_1.shape[0]):
        n1 = texts_1[i].sum()
        n2 = texts_2[i].sum()
        line1 = next(f1)
        line2 = next(f2)
        if n1 == n2:
            diff = np.abs(texts_1[i] - texts_2[i]).sum()
            if diff != 0:
                n_diff_order += 1
                print("tag order diff:", i, diff)
            continue

        print("\n--- diff:", i, n1, n2)
        n_diff_card += 1
        tags1 = set([_t for _t in line1.split()[1:] if _t in tag_id])
        line2 = list(map(int, line2.split()))
        tags2 = set([id_tag[i] for i in range(len(line2)) if line2[i] > 0])
        print("tags 1:", sorted(list(tags1)))
        print("tags 2:", sorted(list(tags2)))

        extra1 = tags1 - tags2
        if len(extra1) > 0:
            print("extra 1:", extra1)
            for k in extra1:
                if k not in tag_id:
                    print("* ERROR:", k, "not it tag_id")
        extra2 = tags2 - tags1
        if len(extra2) > 0:
            print("extra 2:", extra2)
            for k in extra2:
                if k not in tag_id:
                    print("* ERROR:", k, "not it tag_id")
print("#tag order mismatch:", n_diff_order)  # 0
print("#tag cardinarity different:", n_diff_card)  # 16
