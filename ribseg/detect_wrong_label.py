import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import nibabel as nib
# sys.path.append("../totalsegmentator")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "totalsegmentator"))
from map_to_binary import class_map

"""
Detect potential RibSegv2 wrong label by comparing with Totalseg prediction.
"""

# map Totalseg rib to RibSegv2 rib
ts2ribseg = {}
for cid, cn in class_map["total"].items():
    if not cn.startswith("rib_"):
        continue
    # rib_left_1, rib_right_12
    _, side, sid = cn.split('_')
    if "left" == side:
        ts2ribseg[cid] = int(sid)
    elif "right" == side:
        ts2ribseg[cid] = int(sid) + 12

TS_TO_RIBSEGV2 = np.zeros(max(ts2ribseg) + 1, dtype=int)
for ts_cid, ribseg_cid in ts2ribseg.items():
    TS_TO_RIBSEGV2[ts_cid] = ribseg_cid

MIN_TS_COVERAGE = 0.05
MIN_TS_MODE_RATIO = 0.50


def tspred_to_ribsegv2(tspred):
    """map Totalseg prediction to RibSegv2 label space
    Only keep rib classes, other map to background (0).
    Args:
        tspred: int[H, W, L], Totalseg prediction, assuming v2 total class set
    """
    j = np.zeros(tspred.shape, dtype=int)
    valid = tspred < len(TS_TO_RIBSEGV2)
    j[valid] = TS_TO_RIBSEGV2[tspred[valid]]

    return j


def load_label_nii(path):
    label_nii = nib.load(path)
    label = label_nii.get_fdata().astype(int, copy=False)
    return label_nii, label


def detect_case(vid, label_path, tspred_path):
    label_nii, label = load_label_nii(os.path.join(label_path, "RibFrac{}-rib-seg.nii.gz".format(vid)))
    tspred_nii, tspred = load_label_nii(os.path.join(tspred_path, "{}-ts_pred-total.nii.gz".format(vid)))
    tspred = tspred_to_ribsegv2(tspred)

    assert label.shape == tspred.shape, "Shape mismatch {}: label ({}) vs. tspred ({})".format(vid, label.shape, tspred.shape)
    meta_warning = None
    if not np.allclose(label_nii.affine, tspred_nii.affine, atol=1e-3):
        meta_warning = "Affine mismatch"

    mismatch = []
    for c in np.unique(label):
        if c == 0:
            continue
        masked_tspred = tspred[label == c]
        n_vox = int(masked_tspred.size)
        masked_tspred_fg = masked_tspred[masked_tspred > 0]
        n_ts_fg = int(masked_tspred_fg.size)
        if n_ts_fg == 0:
            continue

        counts = np.bincount(masked_tspred_fg, minlength=25)
        tspred_mode = int(counts.argmax())
        coverage = n_ts_fg / n_vox
        mode_ratio = int(counts[tspred_mode]) / n_ts_fg

        # Background means TotalSegmentator missed part of the rib, not that the
        # RibSeg label is wrong. Only flag labels with enough non-background TS
        # evidence, and enough agreement among that evidence.
        if coverage < MIN_TS_COVERAGE or mode_ratio < MIN_TS_MODE_RATIO:
            continue
        if tspred_mode != c:
            mismatch.append({
                "ribseg_label": int(c),
                "totalseg_mode": tspred_mode,
                "totalseg_coverage": round(coverage, 4),
                "totalseg_mode_ratio": round(mode_ratio, 4),
            })

    return vid, mismatch, meta_warning


if "__main__" == __name__:
    data_root = os.path.expanduser("~/data/ribseg")
    label_path = os.path.join(data_root, "ribseg_v2", "seg")
    tspred_path = os.path.join(data_root, "totalseg_pred", "raw")
    workers = max(1, (os.cpu_count() or 2) // 2)

    ignore = (452, 485, 490)
    vids = [vid for vid in range(1, 660 + 1) if vid not in ignore]
    n = len(vids)

    # potential wrong label: {vid: [(ribsegv2_cid, ts_pred_cid)]}
    mismatch = {}
    with ProcessPoolExecutor(max_workers=workers) as executor, \
        open("potential_wrong_label.json", "a") as f:
        f.write(json.dumps({"time": time.asctime()}) + "\n")
        future_to_vid = {
            executor.submit(detect_case, vid, label_path, tspred_path): vid
            for vid in vids
        }
        for n_done, future in enumerate(as_completed(future_to_vid), start=1):
            vid, vid_mismatch, meta_warning = future.result()
            if meta_warning is not None:
                f.write(json.dumps({vid: {"warning": meta_warning}}) + "\n")
                print(vid, meta_warning)
            if vid_mismatch:
                mismatch[vid] = vid_mismatch
                f.write(json.dumps({vid: vid_mismatch}) + "\n")
                print(vid, vid_mismatch)

            print("{}/{}".format(n_done, n), end="\r")

    mismatch = {vid: mismatch[vid] for vid in sorted(mismatch)}
