import os
import common_utils
from config import config
import numpy as np
from ogb.lsc import PCQM4Mv2Evaluator
import pandas as pd

files = [
    # 0.0787
    ('models_valid/final0_dropout_decay1_0.97_h32_hidden256_fastedge_inter8_bondreverse_noxyz_l24_cont2/pred_057.npy', 1),
    # 0.0787
    ('models_valid/final0_dropout_decay1_0.97_h32_hidden256_fastedge_inter8_bondreverse_noxyz_l30/pred_075.npy', 1),
    # 0.0789
    ('models_valid/final10_dropout_decay1_0.97_h16_hidden256_fastedge_inter8_bondreverse_noxyz_l24/pred_075.npy', 1),
    # 0.07897
    ('models_valid/final19_dropout_decay1_0.97_h32_hidden256_fastedge_inter8_bondreverse_noxyz_l30/pred_081.npy', 1),
    # 0.0799
    ('models_valid/final29_dropout_decay1_0.97_h32_nh32_hidden256_fastedge_inter8_bondreverse_noxyz_l24/pred_078.npy', 1),
    # 0.07897
    ('models_valid/final39_dropout_decay1_0.97_h16s16_hidden256_fastedge_nodiag_l30/pred_078.npy', 1),
    # 0.07877
    ('models_valid/final49_dropout_decay1_0.97_h16s16_hidden256_fastedge_l24/pred_085.npy', 1),
    # 0.0801
    ('models_valid/final24_dropout_decay1_0.97_h16s16_hidden256_fastedge_l24/pred_092.npy', 1),
    # valid_0.0811
    ('models_valid/dropout0.1_decay1_0.97_h32s32_hidden256_fastedge_usepredictedgenormcls_ft_train_valid_l24/pred_023.npy', 3),
]

if __name__ == '__main__':
    target_split = 'test-dev'
    data_path = config['middle_data_path']
    idx_split = common_utils.load_obj(os.path.join(data_path, 'idx_split.pkl'))
    y = pd.read_csv('{0}/pcqm4m-v2/raw/data.csv.gz'.format(config['ogb_data_path']))['homolumogap'].values.flatten()

    score_list = []
    score_list_train = []
    num = 0
    for f, n in files:
        scores = np.load(f)
        score_list.append(scores[idx_split[target_split]])
        score_list_train.append(scores[idx_split['train']] * n)
        num += n
        pass

    scores = np.sum(np.stack(score_list, axis=0), axis=0) / num
    scores_train = np.sum(np.stack(score_list_train, axis=0), axis=0) / num
    print(np.abs(y[idx_split['train']] - scores_train).mean())

    input_dict = {'y_pred': scores}


    evaluator = PCQM4Mv2Evaluator()
    evaluator.save_test_submission(input_dict = input_dict, dir_path = '.', mode = 'test-dev')
    pass