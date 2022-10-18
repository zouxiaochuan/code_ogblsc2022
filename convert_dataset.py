import ogb.lsc
import os
import pandas as pd
import common_utils
import features
import pickle
import multiprocessing as mp
from tqdm import tqdm


def process_func(param):
    i, s, y, folder = param

    graph = features.smiles2graph(s)
    
    filename = os.path.join(folder, format(i//1000, '04d'), format(i, '07d') + '.pkl')
    common_utils.save_obj((graph, y), filename)
    # return graph, y
    return

def convert(ogb_data_path, dst_path):
    '''convert original dataset to dataset that we use for training
    '''

    origin_dataset = ogb.lsc.PCQM4Mv2Dataset(ogb_data_path, only_smiles=True)
    idx_split = origin_dataset.get_idx_split()

    csvfile = os.path.join(origin_dataset.folder, 'raw', 'data.csv.gz')
    df = pd.read_csv(csvfile)
    smiles = df['smiles'].values
    labels = df['homolumogap'].values

    dst_data_path = os.path.join(dst_path, 'data')
    os.makedirs(dst_data_path, exist_ok=True)
    common_utils.save_obj(idx_split, os.path.join(dst_path, 'idx_split.pkl'))

    for i in range(0, len(smiles), 1000):
        os.makedirs(os.path.join(dst_data_path, format(i//1000, '04d')), exist_ok=True)
        pass

    process_params = [(i, s, y, dst_data_path) for i, (s, y) in enumerate(zip(smiles, labels))]

    pool = mp.Pool(processes=72)
    results = list(pool.imap_unordered(process_func, tqdm(process_params)))

    # datas = []
    # for graph, y in results:
    #     datas.append((graph, y))
    #     pass
    pool.close()

    # common_utils.save_obj(datas, os.path.join(dst_path, 'data.pkl'))


    
    

if __name__ == '__main__':
    convert('../ogblsc_data', os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m'))
    pass