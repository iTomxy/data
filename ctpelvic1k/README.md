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

MSD-T10 (sub-dataset 3) can also be downloaded from the [Medical Segmentation Decathlon](../msd.md) [project page](http://medicaldecathlon.com/dataaws/).

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

See [*test-data-subsets.ipynb*](./test-data-subsets.ipynb) for initial exploration of those subsets and annotations.

# References

1. (IJCARS 2021) Deep learning to segment pelvic bones: large-scale CT datasets and baseline models - [paper](https://link.springer.com/article/10.1007/s11548-021-02363-8), [github](https://github.com/MIRACLE-Center/CTPelvic1K), [paper with code](https://paperswithcode.com/dataset/ctpelvic1k)
2. [How to download ABDOMEN and CERVIX datasets #19](https://github.com/MIRACLE-Center/CTPelvic1K/issues/19)
3. [Installing Synapse API Clients](https://help.synapse.org/docs/Installing-Synapse-API-Clients.1985249668.html)
