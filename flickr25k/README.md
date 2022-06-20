# MIR-Flickr25k

中文 blog 见 [1]。

MIR-Flickr25k [2,3] is a multi-label dataset,
containing 25,000 samples belonging to 24 categories.
Each image is accompanied with a set of tags,
which can be used as text modality data.
In total,
there are 1,386 tags which occur in at least 20 images.

A processed version is provided at [4].
It's a result of data cleaning,
with resultant 20,015 data.

# Preparation

Files can be download from [5].
We need [mirflickr25k.zip](http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip)
and [mirflickr25k_annotations_v080.zip](http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip)
here.
They are extracted into *mirflickr/* and *mirflickr25k_annotations_v080/*, respectively.

- Images are contained in *mirflickr/*,
with files' name in the format `im<ID>.jpg`.

- *mirflickr/doc/common_tags.txt* shows the 1,386 tags mentioned above
and their occuring frequency.
It's used to determin the tag order.

- The txt files under *mirflickr/meta/tags/* are the processed
(lowercased, spaces removed, *etc.*)
tags of each images.
They are used to process the text modality data.

- The txt files under *mirflickr25k_annotations_v080/* are the annotation files,
named after the class name,
each indicating which data belong to that class.
Note that there are some *duplicated* files named *\<CLASS_NAME\>_r1.txt*.

# Order

The class order is the alphabetical order of the files under *mirflickr25k_annotations_v080/*,
while the tag order is determined by *mirflickr/doc/common_tags.txt*.

# Label

About the number of classes, there are two opinions:

1. There are 24 classes, as in DCMH [4].
This is the result of ignoring those *\*_r1.txt* files.
Actually by comparing those *\<CLASS_NAME\>_r1.txt* and their corresponding
*\<CLASS_NAME\>.txt*,
one can see that the data IDs listed in the former are contained in the latter, too.
That's why I call them the *duplicated* ones.

2. There are 38 classes, as in [6].
That's the result of treating *\<CLASS_NAME\>_r1.txt* and *\<CLASS_NAME\>.txt* as different classes.

Here, we adopt the same setting as DCMH.

# Clean Data

[@coasxu](https://me.csdn.net/weixin_44633882) pointed out that
after filtering out those data with no tags or labels,
we can get 20,015 data,
the same as DCMH [4].


# Shared Files

- [Baidu cloud drive](https://pan.baidu.com/s/19Zud5NQRKQRdcpGGJtpKjg)
- [Kaggle](https://www.kaggle.com/dataset/e593768f204b802f95db5af3f7258e64ad2fe696d2e6d09258eb03509292ece0)

# References

1. [MIR-Flickr25K数据集预处理](https://blog.csdn.net/HackerTom/article/details/98477506)
2. [The MIRFLICKR Retrieval Evaluation](https://press.liacs.nl/mirflickr/mirflickr.pdf)
3. [The MIRFLICKR Retrieval Evaluation](https://press.liacs.nl/mirflickr/)
4. [jiangqy/DCMH-CVPR2017](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_tensorflow/DCMH_tensorflow)
5. [MIRFLICKR Download](https://press.liacs.nl/mirflickr/mirdownload.html)
6. [TSCVT 2017 | SSDH: Semi-Supervised Deep Hashing for Large Scale Image Retrieval](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8101524)
