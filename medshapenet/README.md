中文网志见 [1]。

MedShapeNet [2] is a medical shape dataset.
It contains 114,075 *.stl* files.

It can be downloaded via the *Download dataset* buttom in the [Download](https://medshapenet-ikim.streamlit.app/Download) page in [2],
or one may first download a *.txt* file of data links first,
named *MedShapeNetDataset.txt*,
which is also attached in this folder,
then download data files using `wget` with command:
```shell
wget -i MedShapeNetDataset.txt
```
I'm also providing a downloading shell script here.
See [*download.sh*](download.sh).

# References

1. [下载MedShapeNet](https://blog.csdn.net/HackerTom/article/details/132598276)
2. [MedShapeNet](https://medshapenet-ikim.streamlit.app/)
