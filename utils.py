import glob
import re
import faiss, pickle, numpy as np, os, cv2, shutil
from PIL import Image
from shapely.geometry import Polygon


def kmeans_cluster(patch_luv,
            k=3,
            n_init=10,
            max_iter=50,
            min_valid_pixels=100, **kwargs):
    x_train = patch_luv.astype(np.float32).reshape(-1, 3)
    x_train = patch_luv[x_train[..., 0] > 0]
    if len(x_train) <= min_valid_pixels:
        return None, None
    kmeans = faiss.Kmeans(d=x_train.shape[1], k=k, niter=max_iter, nredo=n_init)
    kmeans.train(x_train)

    dist, index = kmeans.index.search(x_train, 1)
    index = index.reshape(patch_luv.shape[:2])
    return kmeans, index


def _get_membrane(patch, **kwargs):
    patch_luv = cv2.cvtColor(patch, cv2.COLOR_RGB2LUV)
    membrane_mask = np.zeros((patch_luv.shape[0], patch_luv.shape[1]), dtype=np.uint8)
    kmeans_model, cluster_index = kmeans_cluster(patch_luv, **kwargs)
    if cluster_index is None:
        return membrane_mask, None, []
    center = kmeans_model.centroids.copy()

    max_l_idx = np.argmax(center[:, 0])
    membrane_candidates = [idx for idx in range(3) if idx != max_l_idx]
    membrane_cluster_idx = -1
    if len(membrane_candidates) > 0:
        # 选取细胞膜区域
        membrane_candidates = [idx for idx in membrane_candidates]
        membrane_candidates.sort(key=lambda i: sks[i])
        membrane_cluster_idx = membrane_candidates[-1]
        membrane_mask = np.array((cluster_index == membrane_cluster_idx) & (patch_luv[..., 0] > 0), dtype=np.uint8) * 255
    return membrane_mask, membrane_cluster_idx
