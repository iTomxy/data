from __future__ import print_function
import codecs
import multiprocessing
import os
import os.path as osp
import pprint
import time
import threading
from pycocotools.coco import COCO
import gensim
from gensim.models import Doc2Vec
import numpy as np
import scipy.io as sio


"""Multi-Threading version of make.text.py"""


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


def prep_text(tid, sentences):
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
    with codecs.open("input.{}.txt".format(tid), "w", "utf-8") as f:
        for s in sentences:
            s = s.strip()  # must <- maybe trailing space
            if '.' != s[-1]:
                s += '.'
            f.write(s + '\n')
    os.system("java edu.stanford.nlp.pipeline.StanfordCoreNLP " \
        "-annotators tokenize,ssplit -outputFormat conll -output.columns word " \
        "-file input.{}.txt > /dev/null 2>&1".format(tid))
    with codecs.open("input.{}.txt.conll".format(tid), "r", "utf-8") as f:
        doc = " ".join([ln.strip().lower() for ln in f.readlines() if ln.strip() != ""])

    return doc


# multi-threading vars
N_THREAD = max(4, multiprocessing.cpu_count() - 2)
results, mutex_res = [], threading.Lock()
meta_index, mutex_mid = 0, threading.Lock()


def run(tid, id_list):
    global results, meta_index, id_map_data, model
    n = len(id_list)
    while True:
        mutex_mid.acquire()
        meta_idx = meta_index
        meta_index += 1
        mutex_mid.release()
        if meta_index > n:  # NOT greater-equal
            break

        old_id = id_list[meta_idx]
        new_id = id_map_data[old_id]
        _annIds = coco_caps.getAnnIds(imgIds=_old_id)
        _anns = coco_caps.loadAnns(_annIds)
        # print(len(anns))
        # pprint.pprint(anns)
        sentences = [_a["caption"] for _a in _anns]
        # pprint.pprint(sentences)
        doc = prep_text(tid, sentences)
        # pprint.pprint(doc)
        vec = model.infer_vector(doc)
        # print(vec.shape)
        mutex_res.acquire()
        results.append((new_id, vec[np.newaxis, :]))
        mutex_res.release()
        if meta_idx % 1000 == 0:
            print(meta_idx, ',', time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())))

    # remove the intermedia output files (when using Stanford CoreNLP)
    for f in ["input.{}.txt".format(tid), "input.{}.txt.conll".format(tid)]:
        if osp.exists(f):
            os.remove(f)


for _split in SPLIT:
    print("---", _split, "---")
    tic = time.time()
    anno_file = osp.join(ANNO_P, "instances_{}2017.json".format(_split))
    caps_file = osp.join(ANNO_P, "captions_{}2017.json".format(_split))
    coco = COCO(anno_file)
    coco_caps = COCO(caps_file)
    id_list = coco.getImgIds()

    meta_index = 0  # reset for each split
    t_list = []
    for tid in xrange(N_THREAD):
        t = threading.Thread(target=run, args=(tid, id_list))
        t_list.append(t)
        t.start()

    for t in t_list:
        t.join()

    del t_list


assert len(results) == N_DATA
texts = sorted(results, key=lambda t: t[0])  # ascending by new ID
print([t[0] for t in texts])
for i in xrange(100):#N_DATA):
    assert texts[i][0] == i, "* order error"
texts = [t[1] for t in texts]
texts = np.vstack(texts).astype(np.float32)
assert texts.shape[0] == N_DATA
print("texts:", texts.shape, texts.dtype)  # (123287, 300) dtype('<f4')
sio.savemat(osp.join(COCO_P, "texts.COCO.d2v-{}d.mat".format(texts.shape[1])), {"texts": texts})
