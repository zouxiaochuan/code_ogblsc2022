import ogb.lsc
from tqdm import tqdm
import os
import common_utils
import multiprocessing as mp
import numpy as np


if __name__ == '__main__':
    ogb_data_path = '../ogblsc_data'
    data_path = os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m/')
    origin_dataset = ogb.lsc.PCQM4Mv2Dataset(ogb_data_path, only_smiles=True)

    y = origin_dataset.labels

    np.save(os.path.join(data_path, 'y.npy'), y)
    pass