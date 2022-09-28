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
    patch_luv_flatten = patch_luv.astype(np.float32).reshape(-1, 3)
    x_train = patch_luv_flatten[patch_luv_flatten[..., 0] > 0]
    if len(x_train) <= min_valid_pixels:
        return None, None
    kmeans = faiss.Kmeans(d=x_train.shape[1], k=k, niter=max_iter, nredo=n_init)
    kmeans.train(x_train)

    dist, index = kmeans.index.search(patch_luv_flatten, 1)
    index = index.reshape(patch_luv.shape[:2])
    return kmeans, index


def transform_rgb_to_hdab(rgb):
    Q3x3Mat = np.array([[1.20008421, -0.68938773, 0.63621423],
                        [0.68520323, 0.02359692, -0.7100268],
                        [-0.91776857, 1.50870556, 0.30181683]], dtype=np.float32)
    rgb[rgb == 0] = 1.0
    ACC = np.log(rgb / 255.0)
    hdab = -ACC.dot(Q3x3Mat)
    return hdab


def compute_membrane(patch, **kwargs):
    patch_luv = cv2.cvtColor(patch, cv2.COLOR_RGB2LUV)
    kmeans_model, cluster_index = kmeans_cluster(patch_luv, **kwargs)
    gray_val, membrane_pixels_frac, lightness = 0, 0, 0
    if cluster_index is None:
        return np.zeros(patch.shape[:2], dtype=np.uint8), (gray_val, membrane_pixels_frac, lightness)
    center = kmeans_model.centroids.copy()

    max_l_idx = np.argmax(center[:, 0])
    hdab = transform_rgb_to_hdab(patch)
    membrane_candidates = [(idx, np.mean(hdab[cluster_index == idx, 1])) for idx in range(3) if idx != max_l_idx]
    membrane_candidates.sort(key=lambda x: -x[1])
    membrane_cluster_idx, gray_val = membrane_candidates[0]
    membrane_mask = np.array(cluster_index == membrane_cluster_idx, dtype=np.uint8)
    membrane_pixels_frac = np.sum(membrane_mask)/(membrane_mask.shape[0]*membrane_mask.shape[1])
    lightness = np.mean(patch_luv[cluster_index == membrane_cluster_idx, 0])
    return membrane_mask, (gray_val, membrane_pixels_frac, lightness)
