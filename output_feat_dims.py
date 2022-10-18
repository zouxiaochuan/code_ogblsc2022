import datasets
from config import config
from tqdm import tqdm
import numpy as np
import torch
import scipy.spatial.distance
import cluster_utils


if __name__ == '__main__':
    dataset = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='train', rotate=False, path_atom_map=None)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=32,
        collate_fn=datasets.collate_fn
    )
    max_values = torch.zeros(16)

    dists = []
    bar = tqdm(loader)
    max_node = 0
    for batch in bar:
        g, y = batch
        # feat = torch.reshape(g['atom_feat_cate'], (-1, g['atom_feat_cate'].shape[-1]))
        # max_values = torch.maximum(torch.max(feat, dim=0)[0], max_values)
        valence = (g['atom_feat_cate'][:, :, 9] + g['atom_feat_cate'][:, :, 10]).sum(dim=-1)
        valence = valence + g['atom_mask'].sum(dim=-1)
        max_ = torch.max(valence).item()
        max_node = max(max_node, max_)
        bar.set_postfix(max_node=max_node)
        pass
    
    bar.close()
    
    cluster_utils.cluster1d(np.concatenate(dists), 0.02)