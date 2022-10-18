import datasets
from config import config
import numpy as np
import os
from tqdm import tqdm


run_id = 'dropout_decay5_0.8_h16_fastedge_inter8'
iepoch = 40


if __name__ == '__main__':
    dataset_train = datasets.SimplePCQM4MDataset(path=config['middle_data_path'], split_name='train', rotate=False)
    dataset_valid = datasets.SimplePCQM4MDataset(path=config['middle_data_path'], split_name='valid', rotate=False)

    # ys_train = []
    # for i in tqdm(range(len(dataset_train))):
    #     _, y = dataset_train[i]
    #     ys_train.append(y)
    #     pass

    ys_valid = []
    for i in tqdm(range(len(dataset_valid))):
        _, y = dataset_valid[i]
        ys_valid.append(y)
        pass

    model_save_path = os.path.join('models_valid', run_id)
    scores = np.load(os.path.join(model_save_path, f'pred_{iepoch:03d}.npy'))

    pred_valid = scores[dataset_train.idx_split['valid']]
    # pred_train = scores[dataset_train.idx_split['train']]

    print(np.mean(np.abs(pred_valid - np.array(ys_valid))))
    # print(np.mean(np.abs(pred_train - np.array(ys_train))))
    pass