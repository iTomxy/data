import os
import os.path as osp
import numpy as np
import scipy.io as sio
import time


P = "G:/dataset/NUSWIDE"

np.random.seed(int(time.time()))
labels = sio.loadmat(osp.join(P, "labels.nuswide-tc21.mat"))["labels"]
clean_id = sio.loadmat(osp.join(P, "clean_id.nuswide.tc21.AllTags1k.mat"))["clean_id"][0]

id_f2c = {}  # full id -> clean id
# with open(osp.join(P, "clean-full-map.nuswide.tc21.AllTags1k.txt"), "r") as f:
#     for line in f:
#         # format: <clean id> <full id>
#         cid, fid = line.strip().split()
#         id_f2c[int(fid)] = int(cid)
# for cid, fid in enumerate(clean_id):
#     assert cid == id_f2c[fid], "* inconsistent"
for cid, fid in enumerate(clean_id):
    id_f2c[fid] = cid

# convert full ID to clean ID
cvt_f2c = lambda full_indices: [id_f2c[fid] for fid in full_indices]

N_FULL, N_CLASS = labels.shape
N_CLEAN = clean_id.shape[0]
TEST_PER = 100  # test set: 100 data per class
TRAIN_PER = 500  # training set: 500 data per class
N_TEST = TEST_PER * N_CLASS
N_TRAIN = TRAIN_PER * N_CLASS

indices = clean_id.tolist()  # sampled based on the clean part
np.random.shuffle(indices)


print("1. ensure at least `TEST_PER` data per class is fulfilled for test set")
cls_sum = np.sum(labels[indices], axis=0)  # calc class cardinality
classes = np.argsort(cls_sum)  # scarce first, then abundant
id_test = []
cnt = np.zeros_like(labels[0], dtype=np.int32)
for cls in classes:
    print("-> {}".format(cls))
    for i in indices:
        if cnt[cls] >= TEST_PER:  # enough for this class
            break
        if labels[i][cls] == 1:
            id_test.append(i)
            cnt += labels[i]
    #print(cnt)
    assert cnt[cls] >= TEST_PER, "* class {} not enough".format(cls)
    indices = list(set(indices) - set(id_test))  # remove the chosen IDs
    np.random.shuffle(indices)
    #print("left:", len(indices))
assert len(set(id_test)) == len(id_test), "* repeated sampling"
#print("cnt:", cnt)
print("#test:", len(id_test))


print("2. similarly, ensure at least `TRAIN_PER` data per class is fulfilled for training set")
# indices = list(set(indices) - set(id_test))  # remove the chosen test ID
# np.random.shuffle(indices)
cls_sum = np.sum(labels[indices], axis=0)  # re-calculate
classes = np.argsort(cls_sum)  # re-calculate
id_train = []
cnt = np.zeros_like(labels[0], dtype=np.int32)  # reset
for cls in classes:
    print("-> {}".format(cls))
    for i in indices:
        if cnt[cls] >= TRAIN_PER:  # enough for this class
            break
        if labels[i][cls] == 1:
            id_train.append(i)
            cnt += labels[i]
    #print(cnt)
    assert cnt[cls] >= TRAIN_PER, "* class {} not enough".format(cls)
    indices = list(set(indices) - set(id_train))
    np.random.shuffle(indices)
    #print("left:", len(indices))
assert len(set(id_train)) == len(id_train), "* repeated sampling"
#print("cnt:", cnt)
print("#train:", len(id_train))


print("3. make up the rest of test/training set")
# indices = list(set(indices) - set(id_train))  # remove the chosen train ID
# np.random.shuffle(indices)
lack_test = N_TEST - len(id_test)
lack_train = N_TRAIN - len(id_train)
print("lack:", lack_test, ",", lack_train)
id_test.extend(indices[:lack_test])
id_train.extend(indices[lack_test: lack_test + lack_train])
print("#total test:", len(id_test))
print("#total train:", len(id_train))


print("4. the unlabeled part")
# unlabeled = all - labeled(training) - query(test)
id_unlabeled = list(set(indices) - set(id_train) - set(id_test))
print("#unlabeled:", len(id_unlabeled))


print("5. retrieval database")
id_ret = id_train + id_unlabeled
print("#retrieval:", len(id_ret))


assert len(set(id_test).intersection(set(id_ret))) == 0, "* data leaky"
assert len(set(id_train).intersection(set(id_unlabeled))) == 0, "* data leaky"
print("test:", len(id_test))
print("train:", len(id_train))
print("unlabeled:", len(id_unlabeled))
print("ret:", len(id_ret))

print("meta ID: based on clean data only")
meta_id_test = cvt_f2c(id_test)
meta_id_train = cvt_f2c(id_train)
meta_id_unlabeled = cvt_f2c(id_unlabeled)
meta_id_ret = cvt_f2c(id_ret)

sio.savemat(osp.join(P, "split.nuswide-tc21.{}pc.{}pc.mat".format(TEST_PER, TRAIN_PER)), {
    # full ID
    "idx_test": np.asarray(id_test).astype(np.int32),
    "idx_labeled": np.asarray(id_train).astype(np.int32),
    "idx_unlabeled": np.asarray(id_unlabeled).astype(np.int32),
    "idx_ret": np.asarray(id_ret).astype(np.int32),
    # meta ID
    "meta_idx_test": np.asarray(meta_id_test).astype(np.int32),
    "meta_idx_labeled": np.asarray(meta_id_train).astype(np.int32),
    "meta_idx_unlabeled": np.asarray(meta_id_unlabeled).astype(np.int32),
    "meta_idx_ret": np.asarray(meta_id_ret).astype(np.int32),
}, do_compression=True)
