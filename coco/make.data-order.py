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

file_list_train = [(f, "train2017") for f in os.listdir(TRAIN_P) if (".jpg" in f)]
file_list_val = [(f, "val2017") for f in os.listdir(VAL_P) if (".jpg" in f)]
file_list = file_list_train + file_list_val
print("#data:", len(file_list))  # 12,3287

id_key = lambda x: int(x[0].split(".jpg")[0])
file_list = sorted(file_list, key=id_key)  # ascending of image ID
# print(file_list[:15])

with open(osp.join(COCO_P, "id-map.COCO.txt"), "w") as f:
    # format: <original id> <image file name> <original subset>
    for dsc_id, (f_name, subset) in enumerate(file_list):
        _original_id = id_key(f_name)
        f.write("{} {} {}\n".format(_original_id, f_name, subset))
        # if i > 5: break
print("DONE")
