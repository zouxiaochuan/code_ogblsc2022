
import numpy as np
import os
import torch
from tqdm import tqdm


def load_clusters(data_path, num=''):
    if len(num) > 0:
        y_clusters = np.load(os.path.join(data_path, f'y_clusters_{num}.npy'))
    else:
        y_clusters = np.load(os.path.join(data_path, 'y_clusters.npy'))
    return y_clusters


def get_nearest_cluster(y, y_clusters):
    centroids = y_clusters[:, 0]
    return torch.argmin(torch.abs(centroids[None, :] - y[:, None]), dim=1)

def get_nearest_cluster_np(y, y_clusters):
    centroids = y_clusters[:, 0]
    return np.argmin(np.abs(centroids[None, :] - y[:, None]), axis=1)


def cluster1d(y, threshold=0.04):
    y = np.sort(y)

    clusters = [[y[0], y[0]*y[0], 1, y[0], y[0]]]

    bar = tqdm(y)
    for yi in bar:
        c = clusters[-1]
        centroid = c[0] / c[2]
        new_centroid = (c[0] + yi) / (c[2] + 1)
        if abs(new_centroid - c[3]) > threshold or abs(new_centroid - yi) > threshold:
            newc = [yi, yi*yi, 1, yi, yi]
            clusters.append(newc)
            bar.set_postfix({'clusters': len(clusters)})
            pass
        else:
            # c[0] += yi
            # c[1] += yi*yi
            # c[2] += 1
            # c[3] = min(c[3], yi)
            # c[4] = max(c[4], yi)
            pass
        
        pass

    bar.close()
    means = [c[0]/c[2] for c in clusters]
    stds = [np.sqrt(c[1]/c[2] - (c[0]/c[2])**2) for c in clusters]
    nums = [c[2] for c in clusters]
    mins = [c[3] for c in clusters]
    maxs = [c[4] for c in clusters]

    return np.array([means, stds, nums, mins, maxs]).T