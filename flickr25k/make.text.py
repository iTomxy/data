import os
import os.path as osp
import numpy as np
import scipy.io as sio


"""process text (tags)
Tag order is determined by `mirflickr/doc/common_tags.txt`.
"""


N_DATA = 25000
P = "G:/flickr-25k.origin"
TEXT_P = osp.join(P, "mirflickr", "meta", "tags")
COM_TAG_F = osp.join(P, "mirflickr", "doc", "common_tags.txt")


print("read common tags")
tag_id = {}
with open(COM_TAG_F, 'r', encoding='utf-8') as f:
    for tid, line in enumerate(f):
        tag, tag_cardinality = line.split()
        tag_id[tag] = tid
N_TAG = len(tag_id)
print("#tags:", N_TAG)  # 1386

fs_tags = [osp.join(TEXT_P, f) for f in os.listdir(TEXT_P)]
assert len(fs_tags) == N_DATA


def get_tags(tag_f):
    """read accompanying tags of a datum"""
    tag_list = []
    with open(tag_f, 'r', encoding='utf-8') as f:
        for line in f:
            a = line.strip()
            if a in tag_id:
                tag_list.append(a)
    return tag_list


texts = np.zeros((N_DATA, N_TAG), dtype=np.uint8)
for f in fs_tags:
    # file name format: `tag<data ID>.txt`
    # where <data ID> is 1-base
    sid = int(f.split('.txt')[0].split('tags')[-1]) - 1  # shift to 0-base
    tag_list = get_tags(f)
    for t in tag_list:
        if t in tag_id:  # among those common tags
            texts[sid][tag_id[t]] = 1
print("texts:", texts.shape, ", cardinality:", texts.sum())
# texts: (25000, 1386) , cardinality: 94281
sio.savemat(osp.join(P, "texts.flickr25k.mat"), {"texts": texts}, do_compression=True)
