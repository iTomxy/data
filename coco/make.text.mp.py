from __future__ import print_function
import codecs
import itertools
from multiprocessing import Lock, Manager, Pool, Process, Queue, Value
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


"""MultiProcessing version of make.text.py
References:
- https://docs.python.org/2.7/library/multiprocessing.html
- https://zhuanlan.zhihu.com/p/166091204
- https://www.journaldev.com/15631/python-multiprocessing-example
"""


class WrapCOCO:
    # def __init__(self, coco, coco_caps, id_map_data, d2v_model):
    def __init__(self, d2v_model):
        # self.coco = coco
        # self.coco_caps = coco_caps
        # self.id_list = coco.getImgIds()
        # self.n_data = len(self.id_list)
        # self.id_map_data = id_map_data
        self.d2v_model = d2v_model

    # def __getitem__(self, old_id):
    #     assert isinstance(old_id, int)
    #     new_id = self.id_map_data[old_id]
    #     _annIds = self.coco_caps.getAnnIds(imgIds=old_id)
    #     _anns = self.coco_caps.loadAnns(_annIds)
    #     sentences = [_a["caption"] for _a in _anns]
    #     return new_id, sentences

    # def make_iter(self, pid, n_process):
    #     batch_size = (self.n_data + n_process - 1) // n_process
    #     for meta_idx in range(pid * batch_size, min((pid + 1) * batch_size, self.n_data)):
    #         _old_id = self.id_list[meta_idx]
    #         _new_id = self.id_map_data[_old_id]
    #         _annIds = self.coco_caps.getAnnIds(imgIds=_old_id)
    #         _anns = self.coco_caps.loadAnns(_annIds)
    #         sentences = [_a["caption"] for _a in _anns]
    #         yield _new_id, sentences


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
    doc = doc.split()

    return doc


def run(pid, name_space, q_data, q_results, lock_data, lock_results):
    # print(pid, ':', id(name_space), id(name_space.d2v_model),
    #     id(q_data), id(q_results), id(lock_data), id(lock_results))
    while not q_data.empty() or 1 != name_space.flag_data_fin:
        lock_data.acquire()
        if q_data.empty():
            flag_empty = True
            if 1 == name_space.flag_data_fin:
                lock_data.release()
                break
            time.sleep(7)
        else:
            flag_empty = False
            new_id, sentences = q_data.get()
            # print("<- get data:", new_id)
        lock_data.release()

        if flag_empty:
            continue

        doc = prep_text(pid, sentences)
        # pprint.pprint(doc)
        name_space.d2v_model.random.seed(name_space.d2v_seed)  # to keep it consistent
        vec = name_space.d2v_model.infer_vector(doc)
        # print(vec.shape)
        lock_results.acquire()
        q_results.put((new_id, vec[np.newaxis, :]))
        # print("-> put result:", new_id)
        name_space.processed_data += 1
        if name_space.processed_data % 1000 == 0:
            print(name_space.processed_data, ',', time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())))
        lock_results.release()

    # remove the intermedia output files (when using Stanford CoreNLP)
    for f in ["input.{}.txt".format(pid), "input.{}.txt.conll".format(pid)]:
        if osp.exists(f):
            os.remove(f)

    print("END:", pid)


if "__main__" == __name__:
    raise Exception("""(2022.7.6)
        This implementation is really slow.
        Please use `make.text.mt.py` instead.""")

    # COCO
    COCO_P = "/home/dataset/COCO"
    ANNO_P = osp.join(COCO_P, "annotations")
    SPLIT = ["val", "train"]

    # doc2vec
    MODEL = "/home/dataset/Doc2Vec/enwiki_dbow/doc2vec.bin"
    D2V_SEED = 0  # keep consistency
    # pre-trained Doc2Vec model
    model = Doc2Vec.load(MODEL)

    id_map_data = {}
    with open(osp.join(COCO_P, "id-map.COCO.txt"), "r") as f:
        for _new_id, line in enumerate(f):
            line = line.strip()
            _old_id, *_ = line.strip().split()
            id_map_data[int(_old_id)] = _new_id
    N_DATA = len(id_map_data)
    print("#data:", N_DATA)  # 123,287

    N_PROCESS = 4
    lock_data, lock_results = Lock(), Lock()
    # flag_data_fin = Value('H', 0)  # `H` for unsigned int
    q_data = Queue()  # (new_id, sentences)
    q_results = Queue()  # (new_id, doc2vec fecture)
    # MyManager.register('WrapCOCO', WrapCOCO)
    manager = Manager()
    name_space = manager.Namespace()
    name_space.d2v_model = model
    name_space.flag_data_fin = 0
    name_space.processed_data = 0
    name_space.d2v_seed = D2V_SEED

    p_list = []
    for pid in xrange(N_PROCESS):
        p = Process(target=run, args=(pid, name_space,
            q_data, q_results, lock_data, lock_results))
        p_list.append(p)
        p.start()

    for _split in SPLIT:
        print("---", _split, "---")
        anno_file = osp.join(ANNO_P, "instances_{}2017.json".format(_split))
        caps_file = osp.join(ANNO_P, "captions_{}2017.json".format(_split))
        coco = COCO(anno_file)
        coco_caps = COCO(caps_file)
        for i, old_id in enumerate(coco.getImgIds()):
            new_id = id_map_data[old_id]
            _annIds = coco_caps.getAnnIds(imgIds=old_id)
            _anns = coco_caps.loadAnns(_annIds)
            sentences = [_a["caption"] for _a in _anns]
            lock_data.acquire()
            q_data.put((new_id, sentences))
            # print("put data:", new_id)
            lock_data.release()

    name_space.flag_data_fin = 1
    for pid, p in enumerate(p_list):
        # print("join:", pid)
        p.join()

    manager.shutdown()

    results = []
    while not q_results.empty():
        results.append(q_results.get())
    print(len(results))
    print(len(results[0]), results[0][0], results[0][1].shape)

    # ascending by new ID: (new id, doc2vec feature)
    texts = sorted(results, key=lambda t: t[0])
    # for i in range(100):#N_DATA):
    #     assert texts[i][0] == i, "* order error"
    texts = [t[1] for t in texts]
    texts = np.vstack(texts).astype(np.float32)
    # assert texts.shape[0] == N_DATA
    print("texts:", texts.shape, texts.dtype)  # (123287, 300) dtype('<f4')
    sio.savemat(osp.join(COCO_P, "texts.COCO.d2v-{}d.mat".format(texts.shape[1])), {"texts": texts})
