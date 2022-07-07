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


"""Compare the one processed by `make.text.mt.py` and `make.text.py`
Use the same environment as `make.text.py`.
"""


# COCO
COCO_P = "/home/dataset/COCO"
ANNO_P = osp.join(COCO_P, "annotations")
SPLIT = ["val", "train"]
# doc2vec
MODEL = "/home/dataset/Doc2Vec/enwiki_dbow/doc2vec.bin"


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


texts_mt = sio.loadmat(osp.join(COCO_P, "texts.COCO.d2v-300d.mat"))["texts"]
print(texts_mt.shape)

for _split in SPLIT:
    print("---", _split, "---")
    anno_file = osp.join(ANNO_P, "instances_{}2017.json".format(_split))
    caps_file = osp.join(ANNO_P, "captions_{}2017.json".format(_split))
    coco = COCO(anno_file)
    coco_caps = COCO(caps_file)

    id_list = coco.getImgIds()
    for _old_id in np.random.choice(id_list, 2, replace=False):
        _new_id = id_map_data[_old_id]
        print(_new_id, id_map_data[_old_id])
        _annIds = coco_caps.getAnnIds(imgIds=_old_id)
        _anns = coco_caps.loadAnns(_annIds)
        # print(len(anns))
        # pprint.pprint(anns)
        sentences = [_a["caption"] for _a in _anns]
        # pprint.pprint(sentences)
        doc = prep_text(sentences)
        # pprint.pprint(doc)
        model.random.seed(0)
        vec = model.infer_vector(doc)
        # print(vec.shape, vec.dtype)

        assert (vec != texts_mt[_new_id]).sum() == 0, "{}, {}".format(_new_id, _old_id)
    #     break
    # break
print("DONE")

for f in ["input.txt", "input.txt.conll"]:
    if osp.exists(f):
        os.remove(f)
