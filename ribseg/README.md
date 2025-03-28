The RibSeg series dataset (v1 & v2) are designed for point-cloud based rib segmentation.
Raw data are CT volume,
point clouds are generated by sieving HU value (>= 200) based on each volume.
The v1 dataset focus on binary rib segmentation,
i.e. foreground vs. background.
The v2 dataset extends v1 to multi-class segmentation
(background vs. rib 1 - 24).
Center line extracttion is also considered.

# Download

## scans

Follow the [Download](https://github.com/M3DV/RibSeg?tab=readme-ov-file#download) section in the github readme:
1. go to the homepage of [MICCAI 2020 RibFrac Challenge: Rib Fracture Detection and Classification](https://ribfrac.grand-challenge.org/),
2. join the challenge,
3. then download CT scans from there.

## v1 annotations

Follow the [Download](https://github.com/M3DV/RibSeg/tree/ribsegv1#download) section of ribsegv1 branch readme:
download [RibSeg_490_nii.zip](https://zenodo.org/records/5336592).

## v2 annotations

Follow "[RibSeg v2 dataset, description document, and annotations as mesh here.](https://github.com/M3DV/RibSeg/tree/ribsegv2?tab=readme-ov-file#ribseg-v2-dataset-description-document-and-annotations-as-mesh-here)"
in the ribsegv2 branch readme:
download [ribseg_v2.zip](https://drive.google.com/file/d/1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm/view).

# Gather v2 Data

This section gathers CT scans and segmentation labels of v2 data into a unified folder respectively for convenient data loading.

Unzip the downloaded CT scans and v2 labels first.
Then arrange them according to [data_prepare.py](https://github.com/M3DV/RibSeg/blob/ribsegv1/data_prepare.py):

```
ribseg/
|- ribfrac/             # CT scans scattered across 4 folders
|  |- ribfrac-train-images-1/Part1/
|  |  |- RibFrac<VOLUME_ID>-image.nii.gz
|  |- ribfrac-train-images-2/Part2/
|  |- ribfrac-val-images/
|  `- ribfrac-test-images/
`- ribseg_v2/
   |- seg/
   |  |- RibFrac<VOLUME_ID>-rib-seg.nii.gz
   `- cl/
```

Then run [link-data.sh](link-data.sh) (modify paths as needed).
The resulting structure will be:

```
ribsegv2/
|- image/
|  |- RibFrac<VOLUME_ID>-image.nii.gz
`- label/
   |- RibFrac<VOLUME_ID>-rib-seg.nii.gz
```

# Unify Orientation and Spacing

As a preprocessing,
[unify_ori_spc.py](unify_ori_spc.py) unifies the orientation and spacing of each volume to a preset value.

# References

1. (MICCAI'21) RibSeg Dataset and Strong Point Cloud
Baselines for Rib Segmentation from CT Scans - [arXiv](https://arxiv.org/abs/2109.09521), [code](https://github.com/M3DV/RibSeg/tree/ribsegv1)

2. (TMI'23) RibSeg v2: A Large-scale Benchmark for Rib Labeling and Anatomical Centerline Extraction - [arXiv](https://arxiv.org/abs/2210.09309), [code](https://github.com/M3DV/RibSeg/tree/ribsegv2)
