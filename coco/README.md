# MS COCO

中文 blog 见 [1]。

MS COCO 2017 [2] contains 123,287 images,
each being compained with 5 sentences as descriptions.
The objects in those images belong to 80 categories,
although the maximum category ID is 90.

Here, the original train/val splitting is ignored.
All data are combined into a united pool.
COCO API [3] is used.
The showcase of this API usage can be found at [4,5].

The sentences are processed as 300-D Doc2Vec [6] vectors as in [7].
This depends on the pretrained model provided in [8] and Stanford CoreNLP [9,10],
which in turns depends on an older version of gensim forked at [11], Python 2 and Java.
Regarding that the COCO API is compatible with Python 2.7,
I use the docker images provided at [12],
which is built with a conda,
and create a virtual environment of Python 2.7 in it.
Also,
Java and Stanford CoreNLP are installed and configured in the docker container.

# Preparation

## Files of MS COCO

Some files need to be downloaded from [13]:

- [2017 Train images [118K/18GB]](http://images.cocodataset.org/zips/train2017.zip)
- [2017 Val images [5K/1GB]](http://images.cocodataset.org/zips/val2017.zip)
- [2017 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

They are extracted to *train2017/*, *val2017/* and *annotations/*, respectively,
all under the folder *COCO/*.

## Files of Doc2Vec Model

Download the pretrained Doc2Vec model provided at [8]:

- [English Wikipedia Skip-Gram (1.4GB)](https://cloudstor.aarnet.edu.au/plus/s/Ss4E5by3Ukj0JuN/download)

and extract to *Doc2Vec/wiki_sg/*.

## Docker Environment

Steps in this section are done **within** the docker container created from [12].
To make it easy,
I provide a Dockerfile for creating such docker image at [14].

### COCO API & gensim

Clone [3] and [9] and install them within the docker container.
Before installing them,
several dependent packages should be installed:

- Cython
- matplotlib
- scipy
- smart_open

Then for COCO API,
simply run the 4 commands in *cocoapi/PythonAPI/Makefile*.
For gensim,
run `python setup.py install` under the cloned folder *gensim/*.

### Java

Download JDK 1.8 from [15],
extract to */usr/local/java/jdk1.8.0_40/*,
and append the following lines in */etc/profile*:

```shell
export JAVA_HOME=/usr/local/java/jdk1.8.0_40
export CLASSPATH=.:${JAVA_HOME}/lib/dt.jar:${JAVA_HOME}/lib/tools.jar
export PATH=$PATH:${JAVA_HOME}/bin
```

### Stanford CoreNLP

Download Stanford CoreNLP from [9] (version 4.4.0 in my case),
extract to */usr/local/stanford-corenlp/stanford-corenlp-4.4.0/*,
and run the follow command to configure:

```shell
for f in `find /usr/local/stanford-corenlp/stanford-corenlp-4.4.0/ -name "*.jar"`; do
    echo "CLASSPATH=\$CLASSPATH:`realpath $f`" >> /etc/profile;
done
```

Some packages may need to be installed to avoid errors:

```shell
# install some needed packages
apt update
apt install -y libsm6 libxext6
# clean apt cache
apt-get clean
apt-get autoclean
apt-get autoremove
# avoid permission problem
mkdir -p /.cache/Python-Eggs
chmod -R 777 /.cache
```

# Order

Those categories already have their integer IDs defined in the annotation files under *annotations/*,
although those IDs are not continuous.
Here,
we redefine a continuous catetory ID according the ascending order of the original one.
As that both the originally split training and validation set contain all the categories,
here we only use the validation set for processing.

For the data order,
noticing that the file name of those images are actually integers
and all those integers are unique,
we treat them as the original data ID,
which are also not continuous.
Likewise,
we redefine continuous data IDs for them according to the ascending order of the original ones.

# Labels

Labels will be processed as multi-hot vectors
based on the *instances_\*2017.json* under *annotations/*.
The data order and category order obey the aforementioned ones.

# Text

The 5 sentences of each image are treated as a document and processed as Doc2Vec feature following [7].
According to the paper [16] and example code provided in [8],
we first use Stanford CoreNLP tokenise and lowercase the sentences,
then extract feature with the pretrained Doc2Vec model.

Note that because I didn't figure out how to use Stanford CoreNLP for lowercasing,
I lowercase the them with the Python built-in `lower()` method of `str`.

See *make.text.py* for details.
To speed up,
use *make.text.mt.py* to enable multi-threading processing,
and use *check.text.py* to check the consistency with sequential processing method.

# Clean Data

Remove the data with empty label.
See *make.clean.py* for details.

# Comparison

[17] used COCO, too.
And they provided their processed data at [18].
After removing the data with empty label,
we find that our processed label are consistent with theirs.
See *check.label.py* for details.

# Shared Files

[Baidu cloud drive](https://pan.baidu.com/s/1G8R0gHNI33vhx3TQukYDBg), with code: `vgvr`

# References

1. [MS COCO 2017数据集预处理](https://blog.csdn.net/HackerTom/article/details/117001560)
2. [COCO](https://cocodataset.org/#home)
3. [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
4. [cocoapi/PythonAPI/pycocoDemo.ipynb](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb)
5. [COCO-stuff用法](https://blog.csdn.net/HackerTom/article/details/114588496)
6. [ICML 2014 | Distributed Representations of Sentences and Documents](https://proceedings.mlr.press/v32/le14.html)
7. [MM 2019 | Separated variational hashing networks for cross-modal retrieval](https://dl.acm.org/doi/10.1145/3343031.3351078)
8. [jhlau/doc2vec](https://github.com/jhlau/doc2vec)
9. [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
10. [ACL 2014 | The Stanford CoreNLP Natural Language Processing Toolkit](https://aclanthology.org/P14-5010/)
11. [jhlau/gensim](https://github.com/jhlau/gensim)
12. [pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime](https://hub.docker.com/layers/pytorch/pytorch/1.4-cuda10.1-cudnn7-runtime/images/sha256-ee783a4c0fccc7317c150450e84579544e171dd01a3f76cf2711262aced85bf7?context=explore)
13. [COCO download](https://cocodataset.org/#download)
14. [iTomxy/ml-template/docker-files/pt1_4-d2v](https://github.com/iTomxy/ml-template/blob/master/docker-files/pt1_4-d2v)
15. [jdk-8u40-linux-x64.gz](https://pan.baidu.com/s/1Z1Z3Vkq5tgHRVuwqYLsaow), with code `g9jb`
16. [An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation](https://aclanthology.org/W16-1609/)
17. [TNNLS 2020 | Transductive Zero-Shot Hashing for Multilabel Image Retrieval](https://ieeexplore.ieee.org/document/9309012)
18. [qinnzou/Zero-Shot-Hashing](qinnzou/Zero-Shot-Hashing)
