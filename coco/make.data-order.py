import os
import os.path as osp


"""discretization of the original file ID
Map the file ID to sequential {0, 1, ..., n},
and record this mapping in `id-map.txt`,
with format `<new id> <original id> <image file name>`.

Note that the new ids are 0-base.
"""


COCO_P = "/home/dataset/COCO"
TRAIN_P = osp.join(COCO_P, "train2017")
VAL_P = osp.join(COCO_P, "val2017")

file_list = [f for f in os.listdir(TRAIN_P) if (".jpg" in f)]
file_list.extend([f for f in os.listdir(VAL_P) if (".jpg" in f)])
print("#data:", len(file_list))  # 12,3287

id_key = lambda x: int(x.split(".jpg")[0])
file_list = sorted(file_list, key=id_key)  # ascending of image ID
# print(file_list[:15])

with open(osp.join(COCO_P, "id-map.COCO.txt"), "w") as f:
    # format: <original id> <image file name>
    for f_name in file_list:
        _original_id = id_key(f_name)
        f.write("{} {}\n".format(_original_id, f_name))
        # if i > 5: break
print("DONE")
