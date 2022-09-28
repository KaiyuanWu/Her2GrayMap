import os, sys, numpy as np, cv2, json, copy
import glob, pickle
from collections import defaultdict

import openslide, pandas as pd
from collections import OrderedDict
import torch
import logging
import random
from mmdet.datasets import DATASETS, CustomDataset

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class WSIDataset():
    def __init__(self, slide_dir, anno_dir="", label_fn="",
                 mode='train'):
        self._slide_dir, self._anno_dir = slide_dir, anno_dir
        self._label_fn, self._mode = label_fn, mode
        self._slides = self._load_slide()
        self._annos = self._load_anno()
        self._lbl_infos = self._load_label()
        self._data_infos = self._build_data_info()
        self._data_idx = 0

    def _load_slide(self):
        # load all WSI from slide_dir
        if not os.path.exists(self._slide_dir):
            return []
        slides = glob.glob("{}/**/*.mrxs".format(self._slide_dir), recursive=True)
        return slides

    def _load_anno(self):
        # load all annotations from anno_dir
        # annotation of each slide has the same basename with the WSI.

        if not os.path.exists(self._anno_dir):
            return {}
        all_annos = {}
        anno_fns = glob.glob("{}/**/*.geojson".format(self._anno_dir), recursive=True)
        logger.info("anno_fns: {}".format(anno_fns))

        for anno_fn in anno_fns:
            anno_bn = os.path.basename(anno_fn)
            anno_bn, _ = os.path.splitext(anno_bn)
            try:
                with open(anno_fn) as f:
                    anno = json.load(f)
                all_annos[anno_bn] = anno
            except Exception as e:
                logger.warning("fail to load {}. error: {}".format(anno_fn, e))
        return all_annos

    def _load_label(self):
        lbl_infos = {}
        label_df = pd.read_excel(self._label_fn)
        for row_idx, row in label_df.iterrows():
            lbl_infos[row['slide_bn']] = {'ihc': row['ihc'], 'fish': row['fish']}
        return lbl_infos

    def _correct_annotation_shift(self, annos, slide_fn):
        slide = openslide.OpenSlide(slide_fn)
        bounds_x, bounds_y = float(slide.properties['openslide.bounds-x']), float(slide.properties['openslide.bounds-y'])
        bounds = np.array([[bounds_x, bounds_y]])
        for anno in annos:
            properties, geometry = anno.get("properties", {}), anno.get("geometry", {})
            geometry_type, coordinates = geometry.get("type", ""), geometry.get("coordinates")
            if geometry_type == 'LineString':
                coordinates = np.array(coordinates) + bounds
                geometry['coordinates'] = coordinates.astype(np.int32).tolist()
            else:
                for poly_idx, poly in enumerate(coordinates):
                    poly = np.array(poly) + bounds
                    coordinates[poly_idx] = poly.astype(np.int32).tolist()
        return annos

    def _build_data_info(self):
        data_infos = []
        for slide_fn in self._slides:
            slide_bn = os.path.basename(slide_fn)
            slide_bn = os.path.splitext(slide_bn)[0]
            sample = {'slide_fn': slide_fn,
                      'slide_bn': slide_bn,
                      'annos': self._annos.get(slide_bn, {}).get("features", []),
                      'lbls': self._lbl_infos.get(slide_bn, {})}
            sample['annos'] = self._correct_annotation_shift(annos=sample['annos'],
                                           slide_fn=sample['slide_fn'])
            if self._mode == 'train':
                if len(sample['lbls']) > 0:
                    data_infos.append(sample)
                else:
                    logger.warning("training slide: {} doesn't have label.".format(slide_fn))
            else:
                data_infos.append(sample)
        return data_infos

    def __len__(self):
        return len(self._data_infos)

    def __iter__(self):
        self._data_idx = 0
        return self

    def __next__(self):
        if self._data_idx >= len(self._data_infos):
            raise StopIteration
        retval = {key: copy.deepcopy(val) for key, val in self._data_infos[self._data_idx].items()}
        retval['slide'] = openslide.OpenSlide(retval['slide_fn'])
        self._data_idx += 1
        return retval


@DATASETS.register_module(force=True)
class HER2Classification(CustomDataset):
    def __init__(self, *args, fold=-1,
                 uniform_sampling=True,
                 keep_labels=['ihc', 'fish'],
                 label_dims=[4, 2], **kwargs):
        self.fold = fold
        self.uniform_sampling = uniform_sampling
        self._cat_ids = defaultdict(list)
        self.keep_labels = keep_labels
        self.label_dims = label_dims
        super(HER2Classification, self).__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        data_df = pd.read_csv(ann_file)
        if self.fold != -1:
            if not self.test_mode:
                data_infos = [{"filename": "{}/{}.pkl".format(self.img_prefix, row['slide_bn']),
                               'ann': {"ihc": int(row["ihc"]),
                                       "fish": int(row['fish'])},
                               "width": 600,
                               "height": 600} for row_idx, row in data_df.iterrows()
                              if row['fold'] != self.fold]
            else:
                data_infos = [{"filename": "{}/{}.pkl".format(self.img_prefix, row['slide_bn']),
                               'ann': {"ihc": int(row["ihc"]),
                                       "fish": int(row['fish'])},
                               "width": 600,
                               "height": 600} for row_idx, row in data_df.iterrows()
                              if row['fold'] == self.fold]
        else:
            data_infos = [{"filename": "{}/{}.pkl".format(self.img_prefix, row['slide_bn']),
                           'ann': {"ihc": int(row["ihc"]),
                                   "fish": int(row['fish'])},
                            "width": 600,
                            "height": 600} for row_idx, row in data_df.iterrows()]
        logger.info("load #{} items from {}".format(len(data_infos), self.ann_file))
        for item_idx, data_item in enumerate(data_infos):
            self._cat_ids[(data_item['ann']['ihc'], data_item['ann']['fish'])].append(item_idx)
        for cat_id in self._cat_ids:
            random.shuffle(self._cat_ids[cat_id])
        return data_infos

    def __getitem__(self, item):
        if self.uniform_sampling:
            num_cats = len(self._cat_ids)
            cats = list(self._cat_ids.keys())
            cat_idx = random.randint(0, num_cats - 1)
            cat_idx = cats[cat_idx]
            num_samples = len(self._cat_ids[cat_idx])
            sample_idx = random.randint(0, num_samples-1)
            item = self._cat_ids[cat_idx][sample_idx]
        item = super(HER2Classification, self).__getitem__(item)
        return item

    def __len__(self):
        if self.uniform_sampling:
            return len(self._cat_ids)*max([len(self._cat_ids[k]) for k in self._cat_ids])
        else:
            return len(self.data_infos)
