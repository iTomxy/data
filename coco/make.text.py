from __future__ import print_function
import codecs
import os
import os.path as osp
import pprint
import time
from pycocotools.coco import COCO
import gensim
from gensim.models import Doc2Vec
import numpy as np
import scipy.io as sio


"""process texts
python 2 needed by `jhlau/doc2vec`, and COCO api CAN work with python 2.7.
So I choose to create a virtual env of python 2.7.
Dependencies:
    matplotlib (COCO api)
    smart_open (gensim)
Docker:
    iTomxy/ml-template/docker-files/pt1.4-d2v.df
References:
    - https://blog.csdn.net/leitouguan8655/article/details/80533293
"""


# COCO
COCO_P = "/home/dataset/COCO"
ANNO_P = osp.join(COCO_P, "annotations")
SPLIT = ["val", "train"]
# doc2vec
MODEL = "/home/dataset/Doc2Vec/enwiki_dbow/doc2vec.bin"
D2V_SEED = 0  # keep consistency


id_map_data = {}
with open(osp.join(COCO_P, "id-map.COCO.txt"), "r") as f:
    for _new_id, line in enumerate(f):
        _old_id, _ = line.strip().split()
        id_map_data[int(_old_id)] = _new_id
N_DATA = len(id_map_data)
print("#data:", N_DATA)  # 123,287

# pre-trained Doc2Vec model
model = Doc2Vec.load(MODEL)


def prep_text(sentences):
    """preprocess sentences into a single document
    Input:
        - sentences: list of str, one per sentence.
    Output:
        - doc: list of a single str (i.e. all sentences in one line), processed document.
    """
    # use gensim.utils.simple_preprocess
    # sentences = [gensim.utils.simple_preprocess(s) for s in sentences]
    # # pprint.pprint(sentences)
    # doc = []
    # for s in sentences:
    #     doc.extend(s)

    # use Stanford CoreNLP
    with codecs.open("input.txt", "w", "utf-8") as f:
        for s in sentences:
            s = s.strip()  # must <- maybe trailing space
            if '.' != s[-1]:
                s += '.'
            f.write(s + '\n')
    os.system("java edu.stanford.nlp.pipeline.StanfordCoreNLP " \
        "-annotators tokenize,ssplit -outputFormat conll -output.columns word " \
        "-file input.txt > /dev/null 2>&1")
    with codecs.open("input.txt.conll", "r", "utf-8") as f:
        doc = " ".join([ln.strip().lower() for ln in f.readlines() if ln.strip() != ""])
    doc = doc.split()

    return doc


texts = []
for _split in SPLIT:
    print("---", _split, "---")
    anno_file = osp.join(ANNO_P, "instances_{}2017.json".format(_split))
    caps_file = osp.join(ANNO_P, "captions_{}2017.json".format(_split))
    coco = COCO(anno_file)
    coco_caps = COCO(caps_file)

    id_list = coco.getImgIds()
    for i, _old_id in enumerate(id_list):
        _new_id = id_map_data[_old_id]
        _annIds = coco_caps.getAnnIds(imgIds=_old_id)
        _anns = coco_caps.loadAnns(_annIds)
        # print(len(anns))
        # pprint.pprint(anns)
        sentences = [_a["caption"] for _a in _anns]
        # pprint.pprint(sentences)
        doc = prep_text(sentences)
        # pprint.pprint(doc)
        model.random.seed(D2V_SEED)  # to keep it consistent
        vec = model.infer_vector(doc)
        # print(vec.shape)
        texts.append(vec[np.newaxis, :])
        if i % 1000 == 0:
            print(i, time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())))
    #     break
    # break

# remove the intermedia output files (when using Stanford CoreNLP)
if osp.exists("input.txt"):
    os.remove("input.txt")
if osp.exists("input.txt.conll"):
    os.remove("input.txt.conll")

texts = np.vstack(texts).astype(np.float32)
print("texts:", texts.shape, texts.dtype)  # (123287, 300) dtype('<f4')
sio.savemat(osp.join(COCO_P, "texts.COCO.d2v-{}d.mat".format(texts.shape[1])), {"texts": texts})
