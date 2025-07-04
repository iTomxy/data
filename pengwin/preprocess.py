import os, argparse, sys, glob
import numpy as np
import SimpleITK as sitk

sys.path.append("../verse")
from complete_label import predict

"""
Unify orientation, predict with pretrained TotalSegmentator model for unlabelled bones, and slice along the IS dimension.
See ../verse/env_totalseg.sh for the environment needed by TotalSegmentator.
"""

def reorient(image, target_orientation='LPS'):
    """Reorient image to a standard orientation.
    Args:
        image: SimpleITK image
        target_orientation: Target orientation code (default: 'LPS')
                          Common options: 'LPS', 'RAS', 'LPI', 'RAI'
    Returns:
        Reoriented SimpleITK image
    """
    # Get current orientation
    current_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )

    # print(f"Current orientation: {current_orientation}")
    # print(f"Target orientation: {target_orientation}")

    # Only reorient if different from target
    if current_orientation == target_orientation:
        return image

    # Create orientation filter
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(target_orientation)

    # Apply reorientation
    reoriented_image = orient_filter.Execute(image)
    # print(f"Reoriented from {current_orientation} to {target_orientation}")
    return reoriented_image


def reorient_data(args):
    """unify volumes' orientation, and put in a single folder"""
    label_path = os.path.join(args.data_path, "PENGWIN_CT_train_labels")
    image_path = os.path.join(args.data_path, "PENGWIN_CT_train_images_part{}")
    save_path = os.path.join(args.data_path, "reorient_{}".format(args.orientation))
    os.makedirs(save_path, exist_ok=True)

    for part in (1, 2):
        ip = image_path.format(part)
        for f in os.listdir(ip):
            vid = f.split('.')[0]

            image = sitk.ReadImage(os.path.join(ip, f))
            label = sitk.ReadImage(os.path.join(label_path, f))

            image = reorient(image, args.orientation)
            label = reorient(label, args.orientation)

            sitk.WriteImage(image, os.path.join(save_path, 'image_{}.nii.gz'.format(vid)))
            sitk.WriteImage(label, os.path.join(save_path, 'label_{}.nii.gz'.format(vid)))
            print(vid, end='\r')


def ts_infer(args):
    """segment with pretrained TotalSegmentator for missing bones (e.g. spine)"""
    src_path = os.path.join(args.data_path, "reorient_{}".format(args.orientation))
    assert os.path.isdir(src_path), "Path to reoriented data does not exist: {}".format(src_path)

    save_path = os.path.join(args.data_path, "ts_pred")
    os.makedirs(save_path, exist_ok=True)
    for f in glob.iglob(os.path.join(src_path, "image_*.nii.gz")):
        vid = os.path.basename(f)[6: -7]
        predict(f, vid, save_path, subtasks=args.ts_tasks)
        print(vid, end='\r')


def slice_data(args):
    """slice along the IS axis"""
    src_path = os.path.join(args.data_path, "reorient_{}".format(args.orientation))
    assert os.path.isdir(src_path), "Path to reoriented data does not exist: {}".format(src_path)

    ts_pred_path = os.path.join(args.data_path, "ts_pred")
    assert os.path.isdir(ts_pred_path), "TotalSegmentator predictions do not exist: {}".format(ts_pred_path)

    save_path = os.path.join(args.data_path, "slice_is")

    for f in glob.iglob(os.path.join(src_path, "image_*.nii.gz")):
        vid = os.path.basename(f)[6: -7]
        save_dir = os.path.join(save_path, vid)
        if os.path.isdir(save_dir):
            continue

        image = sitk.ReadImage(f)
        label = sitk.ReadImage(os.path.join(src_path, "label_{}.nii.gz").format(vid))
        # axis order changes from LPS to SLP (or SPL?) when converting to numpy
        ts_preds = {
            t: sitk.ReadImage(os.path.join(ts_pred_path, "{}-{}.nii.gz".format(vid, t)))
            for t in args.ts_tasks
        }

        image_np = sitk.GetArrayFromImage(image)
        label_np = sitk.GetArrayFromImage(label)
        ts_preds = {k: sitk.GetArrayFromImage(v) for k, v in ts_preds.items()}
        assert image_np.shape == label_np.shape
        for k, v in ts_preds.items():
            assert image_np.shape == v.shape, "Shape mismatch: image ({}) vs. {} ({})".format(
                image_np.shape, k, v.shape)
        # print(image.shape)

        tmp_dir = save_dir + ".tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        for i in range(image_np.shape[0]):
            _dict = dict(
                spacing=image.GetSpacing(),
                image=image_np[i],
                label=label_np[i]
            )
            for k, v in ts_preds.items():
                _dict[k] = v[i]
            np.savez_compressed(
                os.path.join(tmp_dir, str(i)),
                **_dict
            )

        os.rename(tmp_dir, save_dir)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str, default="~/sd10t/pengwin")
    parser.add_argument('-o', '--orientation', type=str, default="LPS")
    parser.add_argument('-t', '--ts-tasks', type=str, nargs='+', default=["total"],
        choices=["total", "appendicular_bones", "vertebrae_body"],
        help="do what types of TotalSegmentator prediction.")
    args = parser.parse_args()

    args.data_path = os.path.expanduser(args.data_path)
    assert os.path.isdir(args.data_path), args.data_path

    reorient_data(args)
    ts_infer(args)
    slice_data(args)
