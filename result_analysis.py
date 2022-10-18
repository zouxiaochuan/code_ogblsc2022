
import numpy as np
import os
import common_utils
import cluster_utils
import pandas as pd

models = [
    ('dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles_noxyz_l24_cont2', 60),
    ('dropout_decay5_0.8_h16_fastedge_inter8', 60),
    # ('dropout_decay5_0.8_h16_fastedge_inter8_bondreverse', 30)
    ('dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles_ext3', 50)
    # ('cls_dropout_decay5_0.8_h16_fastedge_inter8', 60),
    
    # ('cls0.02_dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles', 30),
]

if __name__ == '__main__':

    scores_list = []
    for name, epoch in models:
        scores = np.load(f'./models_valid/{name}/pred_{epoch:03d}.npy')
        scores_list.append(scores)
        pass

    scores = np.mean(np.stack(scores_list, axis=1), axis=1).astype('float32')
    data_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')
    origin_data_path = '../ogblsc/pcqm4m-v2/raw/data.csv.gz'
    smiles = pd.read_csv(origin_data_path)['smiles'].values

    y = np.load(os.path.join(data_path, 'y.npy'))
    idx_split = common_utils.load_obj(os.path.join(data_path, 'idx_split.pkl'))

    print(np.abs(scores - y)[idx_split['valid']].mean())
    print(np.abs(scores - y)[idx_split['train']].mean())
    for score in scores_list:
        print(np.abs(y-score)[idx_split['valid']].mean())
        print(np.abs(y-score)[idx_split['train']].mean())
        pass

    # scores_list[1] = scores
    scores_diff = np.abs(scores_list[0]-scores_list[1])[idx_split['valid']]
    scores_diff_train = np.abs(scores_list[0]-scores_list[1])[idx_split['train']]

    print(scores_diff.mean())
    print(scores_diff_train.mean())

    print(np.sum(scores_diff>0.04))
    print(np.sum(scores_diff<=0.04))

    print(np.sum(scores_diff_train>0.04))
    print(np.sum(scores_diff_train<=0.04))

    print(np.abs(scores_list[0] - y)[idx_split['valid']][scores_diff>0.04].mean())
    print(np.abs(scores_list[0] - y)[idx_split['valid']][scores_diff<=0.04].mean())

    print(np.abs(scores_list[0] - y)[idx_split['train']][scores_diff_train>0.04].mean())
    print(np.abs(scores_list[0] - y)[idx_split['train']][scores_diff_train<=0.04].mean())

    print(np.std(y[idx_split['valid']][scores_diff>0.04]))
    print(np.std(y[idx_split['valid']][scores_diff<=0.04]))

    valid_loss = np.abs(scores - y)[idx_split['valid']]
    print(np.mean([len(smiles[idx_split['valid'][idx]]) for idx in np.argwhere(valid_loss>0.04).flatten()]))
    print(np.mean([len(smiles[idx]) for idx in idx_split['valid']]))
    pass