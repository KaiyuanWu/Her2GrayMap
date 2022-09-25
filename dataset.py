import os, sys, numpy as np, cv2, json, copy
import glob, pickle
from shapely.geometry import Polygon
import openslide, pandas as pd
from collections import ChainMap
import random

class WSIDataset():
    def __init__(self, slide_dir, anno_dir="", label_fn=""):
        self._slide_dir = slide_dir
        self._anno_dir = anno_dir
        self._label_fn = label_fn

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
        anno_fns = glob.glob("{}/**/*.geojson".format(self._anno_dir))

        for anno_fn in anno_fns:
            anno_bn = os.path.basename(anno_fn)
            anno_bn, _ = os.path.splitext(anno_bn)
            try:
                with open(anno_fn) as f:
                    anno = json.load(f)
                all_annos[anno_bn] = anno
            except Exception as e:
                print("fail to load {}. error: {}".format(anno_fn, e))

    def _load_label(self):
        label_infos = {}
        label_df = pd.read_excel(self._label_fn)
        for row_idx, row in label_df.iterrows():
            pass
