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

    diffs = []
    bar = tqdm(loader)
    for batch in bar:
        g, y = batch
        
        xyz = g['xyz'].to('cuda:0')
        dist_gt = torch.cdist(xyz, xyz, p=2)
        dist_pred = g['predict_pair_dist'].squeeze(-1)
        dist_pred = dist_pred.to('cuda:0')
        mask = g['atom_mask'].to('cuda:0')
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        diff = (dist_gt - dist_pred).abs() / (dist_gt + 1e-12)
        diff = (diff * mask).sum() / mask.sum()
        diffs.append(diff.item())
        bar.set_postfix({'dist_diff': np.mean(diffs)}) 
        
    bar.close()
    return diffs
    pass

if __name__ == '__main__':
    load_dist('data2')