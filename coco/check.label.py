import os.path as osp
import numpy as np
import scipy.io as sio


"""compare my processed labels to (GitHub) qinnzou/Zero-Shot-Hashing
References:
- their github repo: https://github.com/qinnzou/Zero-Shot-Hashing
- their labels: https://github.com/qinnzou/Zero-Shot-Hashing/tree/master/data/coco
"""


MY_P = "G:\\dataset\\COCO\\2017"
ZQ_P = "F:\\codes\\Zero-Shot-Hashing\\data\\coco"


print("Qin's labels")
L_train = sio.loadmat(osp.join(ZQ_P, "train_label.mat"))["train"]
L_val = sio.loadmat(osp.join(ZQ_P, "val_label.mat"))["val"]
L_test = sio.loadmat(osp.join(ZQ_P, "test_label.mat"))["test"]
print(L_train.shape, L_val.shape, L_test.shape)  # (86291, 80) (31983, 80) (5000, 80)
print("train zero:", (0 == L_train.sum(1)).sum())  # 756
print("val zero:", (0 == L_val.sum(1)).sum())  # 258
print("test zero:", (0 == L_test.sum(1)).sum())  # 42
L_zq = np.vstack([L_train, L_val, L_test])
print("L_zq:", L_zq.shape)  # (123274, 80)
rs_zq = L_zq.sum(1)  # row sum

print("my labels")
L_my = sio.loadmat(osp.join(MY_P, "labels.COCO.mat"))["labels"]
rs_my = L_my.sum(1)  # row sum
print("#zero-label:", (0 == rs_my).sum())  # 1069
print("#total:", L_my.shape[0])  # 123287

print("both filter out the data with empty labels")
non_zero_zq = (rs_zq > 0)
L_zq_nz = L_zq[non_zero_zq]
print("#ZQ's non-zero:", L_zq_nz.shape)  # (122218, 80)
clean_id = sio.loadmat(osp.join(MY_P, "clean_id.COCO.mat"))["clean_id"][0]
L_my_nz = L_my[clean_id]
print("#my non-zero:", L_my_nz.shape)  # (122218, 80)


print("compare sorted row sum ")
rs_sort_zq = np.sort(L_zq_nz.sum(1))
rs_sort_my = np.sort(L_my_nz.sum(1))
print("diff:", (rs_sort_zq != rs_sort_my).sum())  # 0

print("- column sum -")
cs_sort_zq = np.sort(L_zq_nz.sum(0))
cs_sort_my = np.sort(L_my_nz.sum(0))
print("diff:", (cs_sort_zq != cs_sort_my).sum())  # 0
