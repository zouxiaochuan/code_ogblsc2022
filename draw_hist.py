import numpy as np
import os
import datasets
from config import config


run_id = 'dropout_decay5_0.8_h16_fastedge_inter8'
iepoch = 40

def cluster(ys):
    cs = [[ys[0], 1, ys[0]]]
    for y in ys:
        c = cs[-1]
        cc = c[0] / c[1]
        newcc = (c[0]+y) / (c[1]+1)
        if np.abs(newcc - c[2]) > 0.04 or np.abs(cc - c[2]) > 0.04:
            newc = [y, 1, y]
            cs.append(newc)
            pass
        else:
            c[0] += y
            c[1] += 1
            pass
        pass

    print(len(cs))
    print([c[0]/c[1] for c in cs[:20]])
    pass


def main():
    preds = np.load(os.path.join('models_valid', run_id, f'pred_{iepoch:03d}.npy'))
    dataset = datasets.SimplePCQM4MDataset(path=config['middle_data_path'], split_name='train', rotate=False)
    data_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')

    y = np.load(os.path.join(data_path, 'y.npy'))
    train_idx = dataset.idx_split['train']
    valid_idx = dataset.idx_split['valid']
    uy = np.unique(y[train_idx])
    cluster(np.sort(np.concatenate((y[train_idx], y[valid_idx]))))

    y = y[valid_idx]
    preds = preds[valid_idx]

    scores = np.abs(y-preds)

    hist, edges = np.histogram(scores, bins=100)
    print(hist)
    print(edges)

    print(np.sum(scores>5))
    print(len(y))
    print(np.sum(scores[(scores>0.04)&(scores<=1.0)])/np.sum(scores))
    print(np.sum(scores[(preds>4)&(preds<8)])/np.sum(scores))
    pass


if __name__ == '__main__':
    main()
    pass