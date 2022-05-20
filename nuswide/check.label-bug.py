import os.path as osp
import glob


P = "G:/dataset/NUSWIDE"
LABEL_P = osp.join(P, "Groundtruth")


for sub_p in ["AllLabels", "TrainTestLabels"]:
    p = osp.join(P, sub_p)
    for fn in glob.glob("{}/*.txt".format(p)):
        with open(fn, "r") as f:
            for ln, line in enumerate(f):
                label = int(line)
                if label not in (0, 1):
                    print("* BUG:", fn, ln, label)
print("DONE")
