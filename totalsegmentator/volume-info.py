import os, json
from collections import defaultdict
import numpy as np
import nibabel as nib

"""
Information of each volume for potential point-cloud based bone segmentation.
"""

def calc_stat(lst, percentages=[], prec=None, scale=None):
    """list of statistics: median, mean, standard error, min, max, percentiles
    It can be useful when you want to know these statistics of a list and
    dump them in a json log/string.
    Input:
        lst: list of number
        percentages: List[float] = [], what percentiles (quantile) to cauculate
        prec: int|None = None, round to which decimal place if it is an int
        scale: int|float|None = None, scale the elements in `lst` if it is an int or float
            Use it when `lst` contains normalised number (i.e. in [0, 1]) and you want to
            present them in percentage (i.e. 0.xyz -> xy.z%)
    """
    if isinstance(scale, (int, float)):
        lst = list(map(lambda x: scale * x, lst))

    ret = {
        "min": float(np.min(lst)),
        "max": float(np.max(lst)),
        "mean": float(np.mean(lst)),
        "std": float(np.std(lst)),
        "median": float(np.median(lst))
    }
    if len(percentages) > 0:
        percentages = [max(1e-7, min(p, 100 - 1e-7)) for p in percentages]
        percentiles = np.percentile(lst, percentages)
        for ptage, ptile in zip(percentages, percentiles):
            ret["p_{}".format(ptage)] = float(ptile)

    if isinstance(prec, int):
        ret = {k: round(v, prec) for k, v in ret.items()}

    return ret


if __name__ == "__main__":
    root = os.path.expanduser("~/data/totalsegmentator")
    min_hu = 200 # to sieve potential bone (edge) voxels
    logger = open("totalseg-volume-info.json", "a")

    size_list, npt_list, sieve_ratio_list = [], [], []
    npt_pc_list = defaultdict(list)
    for vid in os.listdir(os.path.join(root, "data")):
        print(vid, end='\r')
        image = nib.load(os.path.join(root, "data", vid, "ct.nii.gz"))
        label = nib.load(os.path.join(root, "data", vid, "comb_label.nii.gz"))
        ori = ''.join(nib.aff2axcodes(image.affine))
        # assert nib.aff2axcodes(label.affine) == ori
        image = image.get_fdata().astype(np.float32)
        label = label.get_fdata().astype(np.int32)

        class_set = np.unique(label).tolist()
        hu_stat = {
            "whole": calc_stat(image, [0.05, 99.5]),
            "fg": calc_stat(image[label > 0], [0.05, 99.5]),
            "bg": calc_stat(image[0 == label], [0.05, 99.5]),
        }

        mask = (image >= min_hu)
        seg = label[mask]

        size = int(image.size)
        npt = int(seg.shape[0])
        sieve_ratio = npt / size
        size_list.append(size)
        npt_list.append(npt)
        sieve_ratio_list.append(sieve_ratio)
        log = {"vid": vid, "orientation": ori, "n_voxel": size, "npt": npt, "sieve_ratio%": round(100 * sieve_ratio, 4), "npt_pc": [], "class_set": class_set, "hu_stat": hu_stat}
        for cid in np.unique(seg):
            card = (cid == seg).sum()
            log["npt_pc"].append([int(cid), int(card)])
            npt_pc_list[cid].append(card)

        logger.write(json.dumps(log) + os.linesep)

    logger.write(json.dumps({"n_voxel": calc_stat(size_list, prec=2)}) + os.linesep)
    logger.write(json.dumps({"npt": calc_stat(npt_list, prec=2)}) + os.linesep)
    logger.write(json.dumps({"sieve_ratio": calc_stat(sieve_ratio_list, prec=4, scale=100)}) + os.linesep)
    for cid, n_ls in npt_pc_list.items():
        logger.write(json.dumps({"npt_{}".format(cid): calc_stat(n_ls, prec=2)}) + os.linesep)

    logger.flush()
    logger.close()
