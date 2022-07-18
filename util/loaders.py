import cv2
import multiprocessing
import numpy as np
import os.path as osp
from PIL import Image
import threading


class LazyImage:
    """mimics np.ndarray, but uses lazy loading"""

    def __init__(self, image_path, image_size=224, n_thread=None):
        """image_size: int"""
        self.image_path = image_path
        self.image_size = image_size
        # used in resizing
        self.lower_half = image_size // 2
        self.upper_half = (image_size + 1) // 2
        self._mutex_put = threading.Lock()
        self._buffer = []
        self.n_thread = n_thread if (n_thread is not None) else \
            max(1, multiprocessing.cpu_count() - 2)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._load_image(index)
        elif isinstance(index, (np.ndarray, list, tuple)):
            if isinstance(index, np.ndarray):
                assert 1 == index.ndim, "* index should be vector"
            if self.n_thread < 2:  # single thread
                return np.vstack([np.expand_dims(self._load_image(i), 0) for i in index])

            self._buffer = []
            batch_size = (len(index) + self.n_thread - 1) // self.n_thread
            t_list = []
            for tid in range(self.n_thread):
                t = threading.Thread(target=self._load_image_mt, args=(
                    index, range(tid * batch_size, min((tid + 1) * batch_size, len(index)))))
                t_list.append(t)
                t.start()
            for t in t_list:
                t.join()
            del t_list
            assert len(self._buffer) == len(index)
            self._buffer = [t[1] for t in sorted(self._buffer, key=lambda _t: _t[0])]
            return np.vstack(self._buffer)

        raise NotImplemented

    def _load_image_mt(self, indices, seg_meta_indices):
        batch_images = [(mid, np.expand_dims(self._load_image(indices[mid]), 0))
            for mid in seg_meta_indices]
        self._mutex_put.acquire()
        self._buffer.extend(batch_images)
        self._mutex_put.release()

    def _load_image(self, full_index):
        """loads single image resizes
        Input:
            - full_index: int, the sample ID
        """
        img_p = self._get_image_path(full_index)
        img = cv2.imread(img_p)#[:, :, ::-1]
        if img is None:
            with Image.open(img_p) as img_f:
                img = np.asarray(img_f)
            if 2 == img.ndim:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)

        # img = Image.open(img_p)
        # xsize, ysize = img.size
        # seldim = min(xsize, ysize)
        # rate = float(self.image_size) / seldim
        # img = img.resize((int(xsize * rate), int(ysize * rate)))
        # nxsize, nysize = img.size
        # cx, cy = nxsize / 2.0, nysize / 2.0
        # box = (cx - self.lower_half, cy - self.lower_half, cx + self.upper_half, cy + self.upper_half)
        # img = img.crop(box)
        # img = img.convert("RGB")  # can deal with grey-scale images
        # img = img.resize((self.iamge_size, self.image_size))
        # img = np.array(img, dtype=np.float32)
        return img  # [H, W, C]

    def _get_image_path(self, full_index):
        """get image path according to sample ID"""
        raise NotImplemented


class ImageF25k(LazyImage):
    def _get_image_path(self, full_index):
        # shift to 1-base
        return osp.join(self.image_path, "im{}.jpg".format(full_index + 1))

    def __len__(self):
        return 25000


class ImageNUS(LazyImage):
    """depends on (github) iTomxy/data/nuswide/make.image.link.py"""
    def _get_image_path(self, full_index):
        # remain 0-base as is
        return osp.join(self.image_path, "{}.jpg".format(full_index))

    def __len__(self):
        return 269648


class ImageCOCO(LazyImage):
    """depends on (github) iTomxy/data/coco/make.image.link.py"""
    def _get_image_path(self, full_index):
        # remain 0-base as is
        return osp.join(self.image_path, "{}.jpg".format(full_index))

    def __len__(self):
        return 123287
