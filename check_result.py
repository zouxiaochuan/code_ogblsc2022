
import numpy as np
import os
import common_utils
import cluster_utils
import pandas as pd

predict_files = [
    'models_valid/final0_dropout_decay1_0.97_h32_hidden256_fastedge_inter8_bondreverse_noxyz_l24_cont2/pred_057.npy',
    'models_valid/final0_dropout_decay1_0.97_h32_hidden256_fastedge_inter8_bondreverse_noxyz_l30/pred_075.npy'
]

split = 0

if __name__ == '__main__':
    scores_list = []
    for predict_file in predict_files:
        scores = np.load(predict_file).flatten()
        scores_list.append(scores)
        pass

    scores = np.mean(np.stack(scores_list, axis=0), axis=0)
    data_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')
    y = pd.read_csv('../ogblsc_data/pcqm4m-v2/raw/data.csv.gz')['homolumogap'].values.flatten()

    idx_split = common_utils.load_obj(os.path.join(data_path, 'idx_split.pkl'))

    idx_train_valid = np.concatenate([idx_split['train'], idx_split['valid']])
    idx_train = idx_train_valid[np.arange(len(idx_train_valid)) % 50 != split]
    idx_valid = idx_train_valid[np.arange(len(idx_train_valid)) % 50 == split]

    print(np.abs(scores - y)[idx_valid].mean())
    print(np.abs(scores - y)[idx_train].mean())
    print(np.abs(scores - y)[idx_split['train'][:1000]].mean()) 
    print(np.abs(scores - y)[idx_split['valid']].mean()) 
    pass