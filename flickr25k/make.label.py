import os
import os.path as osp
import numpy as np
import scipy.io as sio


"""process label
Class order is the alphabetical order of the name of those files
under `mirflickr25k_annotations_v080/`.
"""


N_DATA = 25000
P = "G:/flickr-25k.origin"
LABEL_P = osp.join(P, 'mirflickr25k_annotations_v080')


fs_lab = [s for s in os.listdir(LABEL_P) if "README" not in s]
fs_lab = [s for s in fs_lab if "_r1" not in s]  # filter out those *_r1.txt

key_lab = lambda s: s.split('.txt')[0]  # file name alphabatical
fs_lab = sorted(fs_lab, key=key_lab)


print("record class order")
with open(osp.join(P, "class-name.flicrk25k.txt"), "w") as f:
    for fn in fs_lab:
        c = key_lab(fn)
        f.write("{}\n".format(c))

fs_lab = [osp.join(LABEL_P, s) for s in fs_lab]
N_CLASS = len(fs_lab)
print("#classes:", N_CLASS)  # 24


def sample_of_lab(lab_f):
    """read annotation file to get the ID list of data belonging to this class
    NOTE: those data ID are 1-base
    """
    samples = []
    with open(lab_f, 'r') as f:
        for line in f:
            sid = int(line)
            samples.append(sid)
    return samples


labels = np.zeros((N_DATA, N_CLASS), dtype=np.uint8)
for i in range(len(fs_lab)):
    samp_ls = sample_of_lab(fs_lab[i])
    for s in samp_ls:
        labels[s - 1][i] = 1  # shift to 0-base
print("labels:", labels.shape, ", cardinality:", labels.sum())
# labels: (25000, 24) , cardinality: 92902
sio.savemat(osp.join(P, "labels.flickr25k.mat"), {"labels": labels}, do_compression=True)
