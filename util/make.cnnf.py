import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch
from util import cnnf
from util.loaders import ImageF25k, ImageNUS, ImageCOCO


DATA_ROOT = "/home/dataset"
PRETRAIN_P = osp.join(DATA_ROOT, "pre-trained")
CNNF_WEIGHT_F = osp.join(PRETRAIN_P, "DCMH.imagenet-vgg-f.mat")
# IMAGE_SIZE = 224


model = cnnf.CNN_F(CNNF_WEIGHT_F).cuda()


@torch.no_grad()
def extract(images, meanpix, batch_size=128):
    """meanpix: [H, W, C]"""
    n = len(images)
    indices = np.arange(n)
    res = []
    for i in range(0, n, batch_size):
        idx = indices[i: i + batch_size]
        img = images[idx].astype(np.float32) - meanpix
        img = torch.from_numpy(img).float().cuda()
        img = img.permute(0, 3, 1, 2)  # -> [n, C, H, W]
        fea = model(img).cpu().numpy()
        res.append(fea)
        print(i)

    res = np.vstack(res)
    print(res.shape, res.mean(), res.min(), res.max())
    return res


print("flickr25k")
P = osp.join(DATA_ROOT, "flickr25k")
mean = sio.loadmat(osp.join(P, "avgpix.flickr25k.py.mat"))["meanpix"]
# (25000, 4096) 0.5188399 0.0 46.786037
features = extract(ImageF25k(osp.join(P, "mirflickr")), mean)
with h5py.File(osp.join(P, "images.flickr25k.cnnf7.h5"), "w") as f:
    f.create_dataset("images", data=features)

print("nuswide")
P = osp.join(DATA_ROOT, "nuswide")
mean = sio.loadmat(osp.join(P, "avgpix.nuswide.py.mat"))["meanpix"]
# (269648, 4096) 0.5332183 0.0 52.547073
features = extract(ImageNUS(osp.join(P, "images")), mean)
with h5py.File(osp.join(P, "images.nuswide.cnnf7.h5"), "w") as f:
    f.create_dataset("images", data=features)

print("COCO")
P = osp.join(DATA_ROOT, "COCO")
mean = sio.loadmat(osp.join(P, "avgpix.COCO.py.mat"))["meanpix"]
# (123287, 4096) 0.5124961 0.0 40.03615
features = extract(ImageCOCO(osp.join(P, "images")), mean)
with h5py.File(osp.join(P, "images.COCO.cnnf7.h5"), "w") as f:
    f.create_dataset("images", data=features)
