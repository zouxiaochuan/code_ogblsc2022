import ogb.lsc
from ogb.utils import smiles2graph

from ogb.lsc import PCQM4Mv2Dataset
import os
import numpy as np
import datasets
from config import config


run_id = 'ce2_dropout_decay5_0.8_h16_fastedge_inter8'
iepoch = 40

pre_idx_file = 'idx_rest_1.npy'

if __name__ == '__main__':
    data_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')
    ce_path = os.path.join(data_path, 'cluster_ensemble')
    os.makedirs(ce_path, exist_ok=True)

    y = np.load(os.path.join(data_path, 'y.npy'))

    model_save_path = os.path.join('models_valid', run_id)
    scores = np.load(os.path.join(model_save_path, f'pred_{iepoch:03d}.npy'))

    if pre_idx_file is not None:
        pre_idx_rest = np.load(os.path.join(ce_path, pre_idx_file))
        pass

    idx_1 = pre_idx_rest & (np.abs(scores - y)<0.025)
    idx_rest = pre_idx_rest & np.logical_not(idx_1)
    
    np.save(os.path.join(ce_path, 'idx_2.npy'), idx_1)
    np.save(os.path.join(ce_path, 'idx_rest_2.npy'), idx_rest)
    pass
