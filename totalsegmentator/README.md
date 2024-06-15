The orientation of all images and labels are `RAS`.

Use [download.sh](download.sh) to download and extract data.
After extraction,
the folder structure will be:
```
totalsegmentator/
|- Totalsegmentator_dataset_v201.zip
|- meta.csv
`- data/
   |- <VOLUME_ID>/
   |  |- ct.nii.gz
   |  `- segmentations/
   |     |- <CLASS_NAME>.nii.gz
```

[classes.json](classes.json) is the class list parsed from the [Class details](https://github.com/wasserth/TotalSegmentator#class-details) section on [1] on June 15, 2024.

See [2] for reorientation with `nibabel`.

# References

1. [wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
2. [Question about orientations #1010](https://github.com/nipy/nibabel/issues/1010)
