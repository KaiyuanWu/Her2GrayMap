import os, sys, pickle, random
import numpy as np
import cv2
from copy import deepcopy
import logging

from mmdet.datasets import PIPELINES

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

@PIPELINES.register_module()
class LoadHer2FromFile(object):
    def __init__(self, keep_labels=[], label_dims=[], with_image=False, generate_bkg_img=-1):
        self.keep_labels = keep_labels
        self.label_dims = label_dims
        self.num_classes = np.sum(self.label_dims)
        self.with_image = with_image
        self.generate_bkg_img = generate_bkg_img
        if self.with_image:
            self.output_ch = 6
        else:
            self.output_ch = 3
        self.bkg_stat = {'mean': np.zeros(self.output_ch),
                         'std': np.ones(self.output_ch)}

    def __call__(self, results):
        filename = results['img_info']['filename']
        with open(filename, "rb") as f:
            img = pickle.load(f)
        img[..., 1] = img[..., 1]*10
        img[..., 2] = img[..., 2]*0.1
        img = img.astype(np.float32)
        if self.with_image:
            img_fn = filename.replace(".pkl", ".png")
            raw_img = cv2.imread(img_fn)
            raw_img = raw_img.astype(np.float32)/255.0
            img = np.concatenate([img, raw_img], axis=-1)
        for c in range(img.shape[-1]):
            self.bkg_stat['mean'][c] = 0.99*self.bkg_stat['mean'][c] + 0.01*np.mean(img[..., c])
            self.bkg_stat['std'][c] = 0.99*self.bkg_stat['std'][c] + 0.01*np.std([img[..., c]])

        if np.random.uniform(0, 1) < self.generate_bkg_img:
            img = np.stack([np.random.randn(img.shape[0], img.shape[1])*self.bkg_stat['std'][i] + self.bkg_stat['mean'][i] for i in range(self.output_ch)], axis=-1).astype(np.float32)
            gt_labels = np.zeros((1, self.num_classes), dtype=np.int64)
        else:
            labels = [results['ann_info'][key] for key in self.keep_labels]
            gt_labels = np.zeros((1, self.num_classes), dtype=np.int64)
            base_idx = 0
            for ix, l in enumerate(labels):
                label_idx = base_idx + l
                gt_labels[0, label_idx] = 1
                base_idx += self.label_dims[ix]
        results['img'] = img
        results['filename'] = filename
        results['gt_labels'] = gt_labels
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}()')
        return repr_str


@PIPELINES.register_module()
class RandomRotate(object):
    def __init__(self, min_angle, max_angle, min_scale=1.0, max_scale=1.0,
                       rotate_cnt_per_sample=1,
                       tgt_size=None):
        """
        Note: tgt_size will not be used anymore. we will use the minimum size that contains the whole of
              the original image content.
        :param min_angle: minimum rotation angle
        :param max_angle: maximum rotation angle
        :param min_scale: minimum resize scale
        :param max_scale: maximum resize scale
        :param tgt_size: output size
        """
        self.min_angle, self.max_angle = min_angle, max_angle
        self.min_scale, self.max_scale = min_scale, max_scale
        self.tgt_size = tgt_size
        self.rotate_cnt_per_sample = rotate_cnt_per_sample
        if self.rotate_cnt_per_sample > 1:
            print("Info rotation will sample #{} each time.".format(self.rotate_cnt_per_sample))

    def __call__(self, results):
        if self.rotate_cnt_per_sample == 1:
            return self.single_call(results)
        results = list(map(lambda i: self.single_call(deepcopy(results)), range(self.rotate_cnt_per_sample)))
        return results

    def single_call(self, results):
        def _get_tgt_size(transform_mat, img_size):
            width, height = img_size
            corner_pnts = np.array([[0, width, width,  0],
                                    [0, 0,     height, height],
                                    [1, 1,     1,      1]], dtype=np.float32)
            new_corner_pnts = transform_mat.dot(corner_pnts)
            x0, y0 = np.min(new_corner_pnts, axis=1)
            x1, y1 = np.max(new_corner_pnts, axis=1)
            return int(x1-x0), int(y1-y0)

        angle = random.uniform(self.min_angle, self.max_angle)
        scale = random.uniform(self.min_scale, self.max_scale)
        transform_mat = None

        for key in results.get("img_fields", ['img']):
            img = results[key]
            height, width, *_ = img.shape
            transform_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            new_width, new_height = _get_tgt_size(transform_mat, (width, height))
            transform_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            transform_mat[:, -1] += np.array([new_width / 2 - width / 2, new_height / 2 - height / 2])
            break
        if transform_mat is None:
            print("Warn: Can not find img_fields in the results. available results keys: {}".format(
                results.keys()
            ))
            return results

        if self.tgt_size is not None:
            new_width, new_height = self.tgt_size

        results['img'] = cv2.warpAffine(results['img'], transform_mat, (new_width, new_height))
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}()')
        return repr_str