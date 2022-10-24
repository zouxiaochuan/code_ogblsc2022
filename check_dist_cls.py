import datasets2 as datasets
from config import config
from tqdm import tqdm
import numpy as np
import torch
import scipy.spatial.distance
import common_utils

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

    bar = tqdm(loader)
    nlist = []
    for batch in bar:
        g, y = batch
        
        xyz = g['xyz'].to('cuda:0')
        dist_gt = torch.cdist(xyz, xyz, p=2)
        mask = g['atom_mask'].to('cuda:0')
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        dist_gt *= mask

        nonzeros = dist_gt[dist_gt>0]
        nlist.append(nonzeros.cpu().numpy())

        pass
        
    bar.close()

    nlist = np.concatenate(nlist)
    values = np.sort(nlist)
    centroids = []

    # thr = 0.2

    # cmin = values[0]
    
    # bar = tqdm(values)
    # for i, v in enumerate(values):
    #     if ((v - cmin) / cmin) > thr:
    #         centroids.append(0.5*(values[i-1] + cmin))
    #         cmin = v
    #     else:
    #         pass

    #     if i % 10000 == 0:
    #         bar.update(10000)
    #         bar.set_postfix({'num': len(centroids)})
    #         pass
    #     pass

    # print(len(centroids))
    
    # centroids = np.array(centroids).astype(np.float32)

    # common_utils.save_obj(centroids, config['middle_data_path'] + '/dist_centroids.pkl')
    num_cls = 16

    interval = len(values) // (num_cls + 1)

    for i in range(num_cls):
        v = values[(i+1)*interval]
        centroids.append(v)
        pass

    centroids = np.array(centroids).astype(np.float32)
    print(centroids)
    common_utils.save_obj(centroids, config['middle_data_path'] + '/dist_centroids.pkl')
    pass

if __name__ == '__main__':
    load_dist('data2')