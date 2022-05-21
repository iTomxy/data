import sys
sys.path.append("..")
import os
import os.path as osp
import numpy as np
import scipy.io as sio
import h5py
import cv2
from PIL import Image
import torch
from util import cnnf


"""CNN-F = VGG-F"""


DATA_P = "/home/dataset"
PRETRAIN_P = osp.join(DATA_P, "pre-trained")
MEANPIX_F = osp.join(PRETRAIN_P, "Mean.h5")  # from GCH
CNNF_WEIGHT_F = osp.join(PRETRAIN_P, "DCMH.imagenet-vgg-f.mat")
IMAGE_SIZE = 224

NUS_P = osp.join(DATA_P, "nuswide")
IMAGE_P = osp.join(NUS_P, "Flickr")
IMAGE_LIST_F = osp.join(NUS_P, "ImageList", "Imagelist.txt")
CLEAN_ID_F = osp.join(NUS_P, "clean_id.nuswide.tc21.AllTags1k.mat")


model = cnnf.CNN_F(CNNF_WEIGHT_F).cuda()
with h5py.File(MEANPIX_F, "r") as f:
    meanpix = f["Mean"][()][np.newaxis, :]  # [1, H, W, C]
    meanpix = torch.from_numpy(meanpix).float().cuda()

# convert path seperator
cvt_sep = lambda p: p.replace('\\/'.replace(os.sep, ''), os.sep)


def load_image(img_p):
    img = cv2.imread(img_p)#[:, :, ::-1]
    if img is None:
        with Image.open(img_p) as _img_f:
            img = np.asarray(_img_f)
        if 2 == img.ndim:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return img[np.newaxis, :]  # [1, H, W, C]


def get_loader(index_set=None, batch_size=128):
    with open(IMAGE_LIST_F, "r") as f:
        _batch, _cnt = [], 0
        for sid, line in enumerate(f):
            if (index_set is not None) and (sid not in index_set):
                continue
            line = cvt_sep(line.strip())
            img_p = osp.join(IMAGE_P, line)
            img = load_image(img_p)
            _batch.append(img)
            _cnt += 1
            if _cnt == batch_size:
                yield np.vstack(_batch)#.astype(np.float32)
                _batch = []
                _cnt = 0

    if len(_batch) > 0:
        yield np.vstack(_batch)


print("extract CNN-F pool5 feature (256, 6, 6)")
clean_id = sio.loadmat(CLEAN_ID_F)["clean_id"][0]
print("#clean data:", clean_id.shape)

fea_list = []
with torch.no_grad():
    # _cnt = 0
    for img_batch in get_loader(clean_id):
        img_batch = torch.from_numpy(img_batch).float().cuda()
        img_batch -= meanpix
        img_batch = img_batch.permute(0, 3, 1, 2)  # -> [n, C, H, W]
        _fea = model(img_batch, "skip-fc").cpu().numpy()
        fea_list.append(_fea)

        # _cnt += _fea.shape[0]
        # print(_cnt, '/', clean_id.shape[0])
        # break
F = np.vstack(fea_list)
print(F.shape)  # [190421, 256, 6, 6]
# `tc21` for label sieving
# `AllTags1k` for text sieving
# `torch`: dimension order differ from TensorFlow
with h5py.File(osp.join(NUS_P, "images.nuswide.tc21.AllTags1k.cnnf5.torch.h5"), "w") as f:
    f.create_dataset("images", data=F)  # 6.6G


print("extract CNN-F 7th layer feature (4096-D)")
fea_list = []
with torch.no_grad():
    # _cnt = 0
    for img_batch in get_loader(None):
        img_batch = torch.from_numpy(img_batch).float().cuda()
        img_batch -= meanpix
        img_batch = img_batch.permute(0, 3, 1, 2)  # -> [n, C, H, W]
        _fea = model(img_batch, None).cpu().numpy()
        fea_list.append(_fea)

        # _cnt += _fea.shape[0]
        # print(_cnt, '/ 269,648')
        # break
F = np.vstack(fea_list)
print(F.shape)  # [190421, 4096]
with h5py.File(osp.join(NUS_P, "images.nuswide.cnnf.h5"), "w") as f:
    f.create_dataset("images", data=F)  # 6.6G
