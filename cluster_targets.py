import os
import numpy as np
import pickle
import common_utils
import cluster_utils


def main():
    data_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')
    idx_split = common_utils.load_obj(os.path.join(data_path, 'idx_split.pkl'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    y_ = y[np.concatenate((idx_split['train'], idx_split['valid']))]

    y_ = y_[y_<12]
    y_clusters = cluster_utils.cluster1d(y_, 0.02)

    print(y_clusters.shape)
    np.save(os.path.join(data_path, 'y_clusters.npy'), y_clusters)

    #
    cluster_idx = cluster_utils.get_nearest_cluster_np(y[idx_split['train']], y_clusters)
    counts = np.zeros(y_clusters.shape[0])
    np.add.at(counts, cluster_idx, 1)
    # counts = np.unique(cluster_idx, return_counts=True)[1]
    print(np.array_str(counts/np.sum(counts), precision=3, suppress_small=True))
    cluster_idx = cluster_utils.get_nearest_cluster_np(y[idx_split['valid']], y_clusters)
    counts = np.zeros(y_clusters.shape[0])
    np.add.at(counts, cluster_idx, 1)
    # counts = np.unique(cluster_idx, return_counts=True)[1]
    print(np.array_str(counts/np.sum(counts), precision=3, suppress_small=True))
    pass


if __name__ == '__main__':
    main()
    pass