import datasets as datasets
import torch.utils.data
from config import config
import models as models
import torch_utils
import torch.optim
import timm.scheduler
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist
import pickle
import common_utils


run_id = 'dropout0.1_decay1_0.97_h32s32_hidden256_classnorm_lesse_train_2'
iepoch = 50
device = 'cuda:2'

def predict():
    
    dataset = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='all', rotate=False, data_path_name='data', load_dist=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=8,
        collate_fn=datasets.collate_fn,
        shuffle=False
    )

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    model = models.MoleculePairDistClassifier(config)
    print('num of parameters: {0}'.format(np.sum([p.numel() for p in model.parameters()])))
    model_save_path = os.path.join('models_valid', run_id)
    sd = torch.load(os.path.join(model_save_path, f'epoch_{iepoch:03d}.pt'), map_location='cpu')
    sd = {k[7:]: v for k, v in sd.items()}
    # print(sd.keys())
    model.load_state_dict(sd)
    model.to(device)
    model_save_path = os.path.join('models_valid', run_id)

    scores_list = []
    model.eval()
    for batch in tqdm(loader):
        graph, y = batch
        graph = torch_utils.batch_to_device(graph, device)
        with torch.no_grad():
            scores = model(
                graph['atom_feat_cate'],
                graph['atom_feat_float'], 
                graph['atom_mask'],
                graph['bond_index'], 
                graph['bond_feat_cate'], 
                graph['bond_feat_float'],
                graph['bond_mask'],
                graph['structure_feat_cate'], 
                graph['structure_feat_float'],
                graph['triplet_feat_cate'])
            pass
        scores = torch.sigmoid(scores)
        scores = scores.detach().cpu().numpy()
        num_atom = graph['atom_mask'].sum(dim=1).detach().cpu().numpy().astype('int64')
        for i, s in enumerate(scores):
            scores_list.append(s[:num_atom[i], :num_atom[i], :])
            pass
        pass


    return scores_list
    pass

def process_fn(param):
    p, i = param
    data_path = os.path.join(config['middle_data_path'], 'data')
    filename = os.path.join(data_path, format(i // 1000, '04d'), format(i, '07d') + '.pkl')

    g, y = common_utils.load_obj(filename)

    g['predict_pair_dist_cls'] = p

    common_utils.save_obj((g, y), filename)
    pass

def main():
    preds = predict()
    pool = mp.Pool()
    params = [(preds[i], i) for i in range(len(preds))]

    list(pool.imap_unordered(process_fn, tqdm(params), chunksize=1024))
    pool.close()
    pass


if __name__ == '__main__':
    main()
    pass