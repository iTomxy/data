import os
import os.path as osp
import numpy as np
import scipy.io as sio
from pycocotools.coco import COCO


"""process labels
Data in both train & val set will be all put together,
with data order determined by `id-map.COCO.txt`
and catetory order by `class-name.COCO.txt`.
"""


COCO_P = "/home/dataset/COCO"
ANNO_P = osp.join(COCO_P, "annotations")
SPLIT = ["val", "train"]


id_map_cls = {}
with open(osp.join(COCO_P, "class-name.COCO.txt"), "r") as f:
    for _new_id, line in enumerate(f):
        _old_id, _ = line.strip().split()
        id_map_cls[int(_old_id)] = _new_id
N_CLASS = len(id_map_cls)
print("#class:", N_CLASS)  # 80

id_map_data = {}
with open(osp.join(COCO_P, "id-map.COCO.txt"), "r") as f:
    for _new_id, line in enumerate(f):
        line = line.strip()
        _old_id, *_ = line.strip().split()
        id_map_data[int(_old_id)] = _new_id
N_DATA = len(id_map_data)
print("#data:", N_DATA)  # 123,287


labels = np.zeros([N_DATA, N_CLASS], dtype=np.uint8)
for _split in SPLIT:
    print("---", _split, "---")
    anno_file = osp.join(ANNO_P, "instances_{}2017.json".format(_split))
    coco = COCO(anno_file)
    id_list = coco.getImgIds()
    for _old_id in id_list:
        _new_id = id_map_data[_old_id]
        _annIds = coco.getAnnIds(imgIds=_old_id)
        _anns = coco.loadAnns(_annIds)
        for _a in _anns:
            _cid = id_map_cls[_a["category_id"]]
            labels[_new_id][_cid] = 1

print("labels:", labels.shape, labels.sum())  # (123287, 80) 357627
sio.savemat(osp.join(COCO_P, "labels.COCO.mat"), {"labels": labels}, do_compression=True)
