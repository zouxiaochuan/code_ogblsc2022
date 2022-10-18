import numpy as np
import os
import sys
import pickle
import scipy.special

run_ids = [
    'dropout_decay5_0.8_h16_fastedge_inter8', 
    'ce1_dropout_decay5_0.8_h16_fastedge_inter8', 
    'ce2_dropout_decay5_0.8_h16_fastedge_inter8'
]

iepoch = 40

middle_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')

def main():
    temp = float(sys.argv[1])
    scores_list = []
    for run_id in run_ids:
        model_save_path = os.path.join('models_valid', run_id)
        scores = np.load(os.path.join(model_save_path, f'pred_{iepoch:03d}.npy'))
        scores_list.append(scores)
        pass

    scores = np.stack(scores_list, axis=1)
    y = np.load(os.path.join(middle_path, 'y.npy'))
    idx_split = pickle.load(open(os.path.join(middle_path, 'idx_split.pkl'), 'rb'))
    train_idx = idx_split['train']
    valid_idx = idx_split['valid']

    l1 = np.abs(scores - y[:, None])
    logits = -l1 / temp
    # exps = np.exp(logits) + 1e-12
    proba = scipy.special.softmax(logits, axis=1)
    pred = np.sum(proba * scores, axis=-1)
    print(np.mean(np.abs(pred-y)[train_idx]))
    print(np.mean(np.abs(pred-y)[valid_idx]))

    np.save(os.path.join(middle_path, 'cluster_ensemble', 'proba.npy'), proba)
    pass


if __name__ == '__main__':
    main()