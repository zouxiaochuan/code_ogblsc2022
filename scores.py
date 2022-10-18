import numpy as np
import os
import pickle
import common_utils

models = [
    # ('dropout_decay5_0.8_h16_fastedge_inter8_bondreverse', 30),
    # ('boost01_dropout_decay5_0.8_h16_fastedge_inter8_bondreverse', 30)
    # ('dropout_decay5_0.8_h16_fastedge_inter8_bondreverse', 30),
    # ('dropout_decay5_0.8_h16_fastedge_inter8', 60),
    # ('cls_dropout_decay5_0.8_h16_fastedge_inter8', 60),
    ('cls0.02_dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles', 30),
    ('boost01_cls0.02_dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles', 30),
    ('boost02_cls0.02_dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles', 30)
]

if __name__ == '__main__':

    scores_list = []
    for name, epoch in models:
        scores = np.load(f'./models_valid/{name}/pred_{epoch:03d}.npy')
        scores_list.append(scores)
        pass

    scores = np.sum(np.stack(scores_list, axis=1), axis=1).astype('float32')
    data_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')

    y = np.load(os.path.join(data_path, 'y.npy'))
    idx_split = common_utils.load_obj(os.path.join(data_path, 'idx_split.pkl'))

    absdiff = np.abs(scores - y)

    print(np.mean(absdiff[idx_split['train']]))
    print(np.mean(absdiff[idx_split['valid']]))

    pass