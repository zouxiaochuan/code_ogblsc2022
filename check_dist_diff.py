import datasets2 as datasets
from config import config
from tqdm import tqdm
import numpy as np
import torch
import scipy.spatial.distance

# torch.multiprocessing.set_sharing_strategy('file_system')

def load_dist(name):
    dataset = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='train', rotate=False, path_atom_map=None, data_path_name=name)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=32,
        collate_fn=datasets.collate_fn
    )

    dists = []
    bar = tqdm(loader)
    for batch in bar:
        g, y = batch
        
        if name == 'data':
            # dist = g['structure_feat_float'].squeeze(-1)
            # dist = torch.cdist(xyz, xyz, p=2)
            dist = g['predict_pair_dist'].squeeze(-1)
        else:
            xyz = g['xyz'].to('cuda:0')
            dist = torch.cdist(xyz, xyz, p=2)
            pass
        num_atom = torch.sum(g['atom_mask'], dim=1).long()
        for i, d in enumerate(dist.cpu().numpy()):
            d = d[:num_atom[i], :num_atom[i]]
            dists.append(d)
            pass
        pass
    bar.close()
    return dists
    pass

if __name__ == '__main__':
    dist_rdkit = load_dist('data')
    dist_sdf = load_dist('data2')

    diffs = []
    bar = tqdm(zip(dist_rdkit, dist_sdf))
    for d1, d2 in bar:
        dist_diff = np.mean(np.abs(d1-d2)/(d2+1e-12))
        
        diffs.append(dist_diff)
        bar.set_postfix({'dist_diff': np.mean(diffs)})
        pass
    bar.close()