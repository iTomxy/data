# NUS-WIDE

My original Chinese blogs can be found at [1,2].

NUS-WIDE [3] is a multi-label dataset,
containing 269,648 samples belonging to 81 categories.

Several files should be downloaded from [2] before the processing here:

- [Groundtruth](https://blog.csdn.net/HackerTom/article/details/Groundtruth),
which contains the label annotations.
After decompression,
there will be two subfolders,
i.e. *AllLabels/* and *TrainTestLabels/*.
We will only use the former one here,
where there are 81 txt files with names in the format of *Labels_\*.txt*.
Each of them corresponds to one category,
containing 269,648 lines of 0/1 indicating whether a sample belongs to the category or not.

- [Tags](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS_WID_Tags.zip),
which will be used as the text modality data.
The downloaded files is *NUS_WID_Tags.zip*,
within which:
    - *Final_Tag_List.txt* is the 5,018 tags mentioned in [4],
    - *TagList1k.txt* is an 1,000-tags subset,
        which matches the statement in [5] in terms of the number of tags,
    - *All_Tags.txt* contains the sample ID and corresponding tags, and
    - *AllTags1k.txt* is a (269648, 1000) 0/1 indicator matrix,
        showing wether a tag is assigned to a sample.

- [Concept List](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/ConceptsList.zip),
from which we get *Concepts81.txt*,
the name list of the 81 categories.

- [Image List](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/ImageList.zip),
which contains *Imagelist.txt*, *TestImagelist.txt* and *TrainImagelist.txt*.
We use *Imagelist.txt* here,
which gives the correspondance between the image IDs and the jpg files.

- [Image Urls](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE-urls.rar),
the download links of the iamges.

The images can also be downloaded from [6,7].
The downloaded images are put in the *Flickr/* directory,
where there are 704 subfolders whose name are corresponding to the ones in *Imagelist.txt*.

# Order

The category order of the label data is determined by *Concepts81.txt*,
the tags order by *TagList1k.txt*,
and the sample order by *Imagelist.txt*.

It's verified that the sample order in *All_Tags.txt* is consistent with that in *Imagelist.txt*,
see *check.sample-order.py*.
And we assume that the sample order in

- those *Labels_\*.txt* files, and
- *AllTags1k.txt*

are consistent with that in *Imagelist.txt*.

# Label

[@Nijiayoudai](https://blog.csdn.net/Nijiayoudai) pointed out that there is a buggy annotation,
that is the 78,372nd line of *Groundtruth/TrainTestLabels/Labels_lake_Train.txt*,
which is `-1130179`,
out of the expected range {0, 1}.
By further checking (see *check.label-bug.py*),
we confirm that that is the only bug in *Groundtruth/TrainTestLabels/*
As the files in the *TrainTestLabels/* subfolder are not used here,
the processed label data are not affected by this bug.

# Text

We follow the setting of DCMH [5],
using the 1000-tags subset as the text modality data.

There are two ways to produce such 1000-D BoW feature, i.e.:

- using *All_Tags.txt* + *TagList1k.txt*, and
- using *AllTags1k.txt*

By comparing the resultant data,
we find that:

- the output of these two methods have **different** number of tags in some of the samples, and
- the tags order or these two methods are the same.

By further comparing them with the text data provided by DCMH [8]
(together with clean ID),
we find that the output of the 2nd method is consistent with that of DCMH.
So we will use the 2nd text data in experiments.

# Image

If the cv2 package is used for image reading,
one should mind that there may be some buggy images,
e.g. *Flickr/albatross/0213_10341804.jpg*,
reading which will result in a `None`.
A more robust reading method can be:

```python
# import numpy as np
# import cv2
# from PIL import Image

img = cv2.imread(img_p)#[:, :, ::-1]
if img is None:  # cv2 failed -> use PIL.Image
    with Image.open(img_p) as img_f:
        img = np.asarray(img_f)
    if 2 == img.ndim:  # lacking channel
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

# TC-10, TC-21

Subsets that only keep the largest 10/21 categories.
See *make.tc.py*.

# Clean Data

Purge those data with empty label or text,
producing clean ID sets and also recording the clean-full ID mappings.
See *make.clean.py*.

# Data Splitting

Besides completely random splitting,
refer to [2] for a splitting principle that used in some works.
The code is put in *make.split.py*.

# Duplication

There are duplicated data (see *check.duplication.py*).
In sum:

- there are only 2-order duplications (i.e. pairs), and
- label inconsistency between those duplicated pairs **exists**.

# Shared Files

- [Baidu cloud drive](https://pan.baidu.com/s/1362XGnPAp5zlL__eF5D_mw), with code: `hf3r`
- [Kaggle](https://www.kaggle.com/dataset/7cbbf047bc9c47b4f2c00e83531d3376ab8887bb0deed2ce2ee1596fe96aa94d)

# References

1. [NUS-WIDE数据集预处理](https://blog.csdn.net/HackerTom/article/details/110092390)
2. [NUS-WIDE数据集划分](https://blog.csdn.net/HackerTom/article/details/104034867)
3. [NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)
4. [NUS-WIDE: A Real-World Web Image Database from National University of Singapore](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/nuswide-civr2009.pdf)
5. (DCMH paper) [Deep Cross-Modal Hashing](https://ieeexplore.ieee.org/document/8099831)
6. [About NUS-WIDE #8](https://github.com/lhmRyan/deep-supervised-hashing-DSH/issues/8) -> [Baidu cloud drive](https://pan.baidu.com/s/1kVl3iSJ), with code: `74qk`
7. [NUS-WIDE数据库在哪里下载？](https://www.zhihu.com/question/50985355/answer/257063493) -> [Baidu cloud drive](https://pan.baidu.com/share/init?surl=kVKfXFx), with code: `hpxg`
8. (DCMH code) [DCMH-CVPR2017/DCMH_tensorflow/DCMH_tensorflow/readme.txt](https://github.com/jiangqy/DCMH-CVPR2017/blob/master/DCMH_tensorflow/DCMH_tensorflow/readme.txt)
