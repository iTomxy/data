To download Abdomen (sub-dataset 1) and Cervix (sub-dataset 5),
see [2],
one should first go to [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)
and join the challenge,
then go back to the dataset links in [1/github] (websites hosted in Synapse).
On clicking the download button in these 2 Synapse dataset websites,
they don't start direct downloading,
but rather add that entry to a download list.
To really start downloading,
one should install the Synapse API following [3],
and then download using the API.

There are 5 classes in the annoataion:
```json
{
    "0": "background",
    "1": "sacrum",
    "2": "right_hip",
    "3": "left_hip",
    "4": "lumbar_vertebra"
}
```
See [MIRACLE-Center/CTPelvic1K/nnunet/dataset_conversion/JstPelvisSegmentation_5label.py](https://github.com/MIRACLE-Center/CTPelvic1K/blob/main/nnunet/dataset_conversion/JstPelvisSegmentation_5label.py#L108) within [1/github] for reference.

# Sub-datasets

## dataset 1 Abdomen

Files in the download website:

- Abdomen.zip
- RawData.zip
- Reg-Training-Testing.zip
- Reg-Training-Training.zip

Note that the content in the last 3 zip files are ALL included in the 1st one,
i.e. *Abdomen.zip*.
So only download the 1st one is enough.

```
Abdomen.zip
|- RawData/
|  |- Testing/img/
|  |  |- img<ID>.nii.gz
|  `- Training/
|     |- img/
|     |  |- img<ID>.nii.gz
|     `- label/
|        |- label<ID>.nii.gz
`- RegData/
   |- Training-Testing/
   |  |- img/
   |  |  |- <testID>/
   |  |  |  |- img<trainID>-<testID>.nii.gz
   |  `- label/
   |     |- <testID>/
   |     |  |- label<trainID>-<testID>.nii.gz
   |- Training-Training/
   |  |- img/
   |  |  |- <trainID_1>/
   |  |  |  |- img<trainID_2>-<trainID_1>.nii.gz
   |  `- label/
   |     |- <trainID_1>/
   |     |  |- img<trainID_2>-<trainID_1>.nii.gz
   `- Training-Training/
```

[1] only use part of it.
The ID of the chosen data can be found in the mapping-back label files in [1/github].

## dataset 3 Colon

Can also be downloaded from the [Medical Segmentation Decathlon](../msd.md) [project page](http://medicaldecathlon.com/dataaws/).

# References

1. (IJCARS 2021) Deep learning to segment pelvic bones: large-scale CT datasets and baseline models - [paper](https://link.springer.com/article/10.1007/s11548-021-02363-8), [github](https://github.com/MIRACLE-Center/CTPelvic1K), [paper with code](https://paperswithcode.com/dataset/ctpelvic1k)
2. [How to download ABDOMEN and CERVIX datasets #19](https://github.com/MIRACLE-Center/CTPelvic1K/issues/19)
3. [Installing Synapse API Clients](https://help.synapse.org/docs/Installing-Synapse-API-Clients.1985249668.html)
