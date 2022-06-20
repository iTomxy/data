import os
import os.path as osp
import numpy as np
import scipy.io as sio
# import gensim
# from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors, Word2Vec


"""extract class embeddings (using GloVe)
NOTE: different from the Google News version.
"""


glove_p = "/home/dataset/GloVe"
which = "840B.300d"  # "6B.300d"  # "42B.300d"
w2v_f = osp.join(glove_p, "w2v.glove.{}.txt".format(which))
w2v = KeyedVectors.load_word2vec_format(w2v_f, binary=False)
# usage: w2v[str] -> numpy array

data_p = "/home/dataset/flickr"
# label_p = osp.join(data_p, "mirflickr25k_annotations_v080")
# test_ls = os.listdir(label_p)
# test_ls = [f for f in test_ls if ("_r1" not in f) and ("README" not in f)]
# N_CLASS = len(test_ls)
# print("#class:", N_CLASS)


cls_emb = []
with open(osp.join(data_p, "class-name.flicrk25k.txt"), "r") as f:
    # now those compounded class names are seperated by space
    for cid, line in enumerate(f):
        c_name = line.strip()
        c_name = c_name.split('_')
        print(c_name, ',', cid)

        emb = np.zeros([300], dtype=np.float32)
        for s in c_name:
            if s not in w2v:
                print("not in vocab:", cid, s)
            else:
                emb += w2v[s]

        cls_emb.append(emb)

cls_emb = np.vstack(cls_emb)
print("class emb:", cls_emb.shape, cls_emb.min(), cls_emb.max(), cls_emb.mean())
# class emb: (24, 300) -3.0838 5.1571 0.00011771765
sio.savemat(osp.join(data_p, "class_emb.glove-{}.flickr25k.mat".format(
    which)), {"class_emb": cls_emb})
