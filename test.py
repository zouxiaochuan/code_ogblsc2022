import ogb.lsc
from ogb.utils import smiles2graph

from ogb.lsc import PCQM4Mv2Dataset
import os
import numpy as np
import datasets
from config import config


run_id = 'dropout_decay5_0.8_h16_fastedge_inter8'
iepoch = 40

run_id2 = 'ce1_dropout_decay5_0.8_h16_fastedge_inter8'
run_id3 = 'ce2_dropout_decay5_0.8_h16_fastedge_inter8'
run_id4 = 'ce3_dropout_decay5_0.8_h16_fastedge_inter8'

data_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')

y = np.load(os.path.join(data_path, 'y.npy'))

model_save_path = os.path.join('models_valid', run_id)
scores = np.load(os.path.join(model_save_path, f'pred_{iepoch:03d}.npy'))

scores1 = np.load(os.path.join('models_valid', run_id2, f'pred_{iepoch:03d}.npy'))
scores2 = np.load(os.path.join('models_valid', run_id3, f'pred_{iepoch:03d}.npy'))
scores3 = np.load(os.path.join('models_valid', run_id4, f'pred_{iepoch:03d}.npy'))

dataset = datasets.SimplePCQM4MDataset(path=config['middle_data_path'], split_name='train', rotate=False)

train_idx = dataset.idx_split['train']
valid_idx = dataset.idx_split['valid']

# y_valid = y['valid_idx']
# scores_valid = scores['valid_idx']


y = y[valid_idx]
scores = scores[valid_idx]
scores1 = scores1[valid_idx]
scores2 = scores2[valid_idx]
scores3 = scores3[valid_idx]

idx0 = np.abs(scores - y)<0.01
idx1 = np.abs(scores1 - y)<0.015
idx2 = np.abs(scores2 - y)<0.025
idx3 = np.abs(scores3 - y)<0.01
idx = np.logical_not(idx0) & np.logical_not(idx1) & np.logical_not(idx2) & idx3
print(np.sum(idx))
idx_rest = np.logical_not(idx0) & np.logical_not(idx1) & np.logical_not(idx2)
print(np.mean(np.abs(scores[idx_rest] - y[idx_rest])))
print(np.mean(np.abs(y - scores3)))
print(np.mean(np.abs(0.333*(scores + scores1 + scores2) - y)))
print(np.mean(np.abs(scores2-scores)))
print(np.sum(np.abs(scores - y)<0.04))