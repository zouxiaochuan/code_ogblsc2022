import ogb.lsc
import os
import pandas as pd
import common_utils
import features
from config import config
import multiprocessing as mp
from tqdm import tqdm
from rdkit import Chem
import numpy as np


def process_func_smiles(param):
    i, s, y, folder = param

    graph = features.smiles2graph(s)
    
    filename = os.path.join(folder, format(i//1000, '04d'), format(i, '07d') + '.pkl')
    common_utils.save_obj((graph, y), filename)
    # return graph, y
    return

def process_func_mol(param):
    i, m, y, folder = param

    graph = features.mol2graph(m)
    
    filename = os.path.join(folder, format(i//1000, '04d'), format(i, '07d') + '.pkl')
    common_utils.save_obj((graph, y), filename)
    # return graph, y
    return

def convert(ogb_data_path, dst_path):
    '''convert original dataset to dataset that we use for training
    '''

    origin_dataset = ogb.lsc.PCQM4Mv2Dataset(ogb_data_path, only_smiles=True)
    idx_split = origin_dataset.get_idx_split()
    sdf_file = os.path.join(ogb_data_path, 'pcqm4m-v2-train.sdf')
    sdf_iter = Chem.SDMolSupplier(sdf_file)

    csvfile = os.path.join(origin_dataset.folder, 'raw', 'data.csv.gz')
    df = pd.read_csv(csvfile)
    smiles = df['smiles'].values
    labels = df['homolumogap'].values

    dst_data_path = os.path.join(dst_path, 'data_sdf')
    os.makedirs(dst_data_path, exist_ok=True)
    common_utils.save_obj(idx_split, os.path.join(dst_path, 'idx_split.pkl'))

    for i in range(0, len(smiles), 1000):
        os.makedirs(os.path.join(dst_data_path, format(i//1000, '04d')), exist_ok=True)
        pass

    # process_params = [(i, s, y, dst_data_path) for i, (s, y) in enumerate(zip(smiles, labels))]

    # process train
    process_params = ((i, sdf_iter[i], labels[i], dst_data_path) for i in range(len(sdf_iter)))

    pool = mp.Pool(processes=36)
    results = list(pool.imap_unordered(process_func_mol, tqdm(process_params)))


    indices_other = np.concatenate(
        (idx_split['valid'], idx_split['test-dev'], idx_split['test-challenge']))
    
    process_params = ((i, smiles[i], labels[i], dst_data_path) for i in indices_other)

    results = list(pool.imap_unordered(process_func_smiles, tqdm(process_params)))
    pool.close()
    
    

if __name__ == '__main__':
    convert(config['ogb_data_path'], config['middle_data_path'])
    pass