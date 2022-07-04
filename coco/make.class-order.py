import os
import os.path as osp
from pycocotools.coco import COCO
import pprint


"""process class order
Record the mapping between tightened/discretized 0-base class ID,
original class ID and class name in `class-name.COCO.txt`,
with format `<original ID> <class name>`.

The class order is consistent to the ascending order of the original IDs.
"""


COCO_P = "/home/dataset/COCO"
ANNO_P = osp.join(COCO_P, "annotations")
SPLIT = ["val", "train"]

for _split in SPLIT:
    print("---", _split, "---")
    anno_file = osp.join(ANNO_P, "instances_{}2017.json".format(_split))
    coco = COCO(anno_file)
    cats = coco.loadCats(coco.getCatIds())
    # print(cats[0])
    cat_list = sorted([(c["id"], c["name"]) for c in cats],
        key=lambda t: t[0])  # ensure ascending
    # pprint.pprint(cat_list)
    with open(osp.join(COCO_P, "class-name.COCO.txt"), "w") as f:
        for old_id, c in cat_list:
            cn = c.replace(" ", "_")
            # format: <original ID> <class name>
            f.write("{} {}\n".format(old_id, cn))

    break  # use val set only is enough
