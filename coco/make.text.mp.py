from __future__ import print_function
import codecs
import itertools
from multiprocessing import Pool
from multiprocessing.managers import BaseManager
import os
import os.path as osp
import pprint
import time
from pycocotools.coco import COCO
import gensim
from gensim.models import Doc2Vec
import numpy as np
import scipy.io as sio


"""MultiProcessing version of make.text.py"""


class WrapCOCO:
    def __init__(self, coco, coco_caps, id_map_data, d2v_model):
        self.coco = coco
        self.coco_caps = coco_caps
        self.id_list = coco.getImgIds()
        self.n_data = len(self.id_list)
        self.id_map_data = id_map_data
        self.d2v_model = d2v_model

    def make_iter(self, pid, n_process):
        batch_size = (self.n_data + n_process - 1) // n_process
        for meta_idx in range(pid * batch_size, min((pid + 1) * batch_size, self.n_data)):
            _old_id = self.id_list[meta_idx]
            _new_id = self.id_map_data[_old_id]
            _annIds = coco_caps.getAnnIds(imgIds=_old_id)
            _anns = coco_caps.loadAnns(_annIds)
            sentences = [_a["caption"] for _a in _anns]
            yield _new_id, sentences


class MyManager(BaseManager):
    pass


def prep_text(pid, sentences):
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
    with codecs.open("input.{}.txt".format(pid), "w", "utf-8") as f:
        for s in sentences:
            s = s.strip()  # must <- maybe trailing space
            if '.' != s[-1]:
                s += '.'
            f.write(s + '\n')
    os.system("java edu.stanford.nlp.pipeline.StanfordCoreNLP " \
        "-annotators tokenize,ssplit -outputFormat conll -output.columns word " \
        "-file input.{}.txt > /dev/null 2>&1".format(pid))
    with codecs.open("input.{}.txt.conll".format(pid), "r", "utf-8") as f:
        doc = " ".join([ln.strip().lower() for ln in f.readlines() if ln.strip() != ""])

    return doc


def run(args):
    pid, wrap_coco, n_process = args
    res = []  # list of (new_id, doc2vec feature)
    for i, (new_id, sentences) in enumerate(wrap_coco.make_iter(pid, n_process)):
        doc = prep_text(pid, sentences)
        # pprint.pprint(doc)
        vec = wrap_coco.d2v_model.infer_vector(doc)
        # print(vec.shape)
        res.append((new_id, vec[np.newaxis, :]))
        if i % 1000 == 0:
            print(pid, ':', i, ',', time.strftime(
                "%Y-%m-%d-%H-%M", time.localtime(time.time())))

    # remove the intermedia output files (when using Stanford CoreNLP)
    for f in ["input.{}.txt".format(pid), "input.{}.txt.conll".format(pid)]:
        if osp.exists(f):
            os.remove(f)

    return res


if "__main__" == __name__:
    # COCO
    COCO_P = "/home/dataset/COCO"
    ANNO_P = osp.join(COCO_P, "annotations")
    SPLIT = ["val", "train"]

    # doc2vec
    MODEL = "/home/dataset/Doc2Vec/enwiki_dbow/doc2vec.bin"
    # pre-trained Doc2Vec model
    model = Doc2Vec.load(MODEL)

    id_map_data = {}
    with open(osp.join(COCO_P, "id-map.COCO.txt"), "r") as f:
        for _new_id, line in enumerate(f):
            line = line.strip()
            _old_id, _ = line.strip().split()
            id_map_data[int(_old_id)] = _new_id
    N_DATA = len(id_map_data)
    print("#data:", N_DATA)  # 123,287

    MyManager.register('WrapCOCO', WrapCOCO)
    manager = MyManager()
    manager.start()

    texts = []
    for _split in SPLIT:
        print("---", _split, "---")
        tic = time.time()
        anno_file = osp.join(ANNO_P, "instances_{}2017.json".format(_split))
        caps_file = osp.join(ANNO_P, "captions_{}2017.json".format(_split))
        coco = COCO(anno_file)
        coco_caps = COCO(caps_file)
        wrap_coco = WrapCOCO(coco, coco_caps, id_map_data, model)

        pool = Pool(4)
        n_process = pool._processes
        res = pool.map_async(run, zip(
            range(n_process), [wrap_coco] * n_process, [n_process] * n_process))
        pool.close()
        res = res.get()
        print("#process:", len(res), "\n#data per process:", len(res[0]), "\ntuple len:", len(res[0][0]),
            "\noriginal ID:", res[0][0][0], "\nd2v feature shape:", res[0][0][1].shape)
        texts.extend(list(itertools.chain(*res)))

        print(_split, ':', time.time() - tic, 's')
        # break

    manager.shutdown()

    # ascending by new ID: (new id, doc2vec feature)
    texts = sorted(texts, key=lambda t: t[0])
    texts = [t[1] for t in texts]
    texts = np.vstack(texts).astype(np.float32)
    assert texts.shape[0] == N_DATA
    print("texts:", texts.shape, texts.dtype)  # (123287, 300) dtype('<f4')
    sio.savemat(osp.join(COCO_P, "texts.COCO.d2v-{}d.mat".format(texts.shape[1])), {"texts": texts})
