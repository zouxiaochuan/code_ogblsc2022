import os
import numpy as np
import pickle
import common_utils
import cluster_utils

previous_preds = [
    ('cls0.02_dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles', 30),
    ('boost01_cls0.02_dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles', 30)
]

def main():
    data_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')

    scores_list = []
    for name, epoch in previous_preds:
        scores = np.load(f'./models_valid/{name}/pred_{epoch:03d}.npy')
        scores_list.append(scores)
        pass

    scores = np.sum(np.stack(scores_list, axis=1), axis=1).astype('float32')

    idx_split = common_utils.load_obj(os.path.join(data_path, 'idx_split.pkl'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    idx_select = np.concatenate((idx_split['train'], idx_split['valid']))
    y_ = y[idx_select]
    scores = scores[idx_select]

    y_ = y_ - scores
    
    y_clusters = cluster_utils.cluster1d(y_, 0.01)

    print(y_clusters.shape)
    np.save(os.path.join(data_path, 'y_clusters_02.npy'), y_clusters)

    pass


if __name__ == '__main__':
    main()
    pass