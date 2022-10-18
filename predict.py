import datasets2 as datasets
import torch.utils.data
from config import config
import models as models
import torch_utils
import torch.optim
import timm.scheduler
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import global_data
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist

run_id = 'dropout_decay1_0.97_h16_hidden256_fastedge_int8_bondreverse_edgenorm_valid-train'
iepoch = 19
device = 'cuda:2'

def main():

    dataset = datasets.SimplePCQM4MDataset(path=config['middle_data_path'], split_name='all', rotate=False, data_path_name='data2')

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn,
        shuffle=False
    )

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    model = models.MoleculeNet(config)
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
            scores, attentions = model(
                graph['atom_feat_cate'],
                graph['atom_feat_float'], 
                graph['atom_mask'],
                graph['bond_index'], 
                graph['bond_feat_cate'], 
                graph['bond_feat_float'],
                graph['bond_mask'],
                graph['structure_feat_cate'], 
                graph['structure_feat_float'],
                graph['triplet_feat_cate'], return_attention=True)
            pass
            scores_list.append(scores.detach().cpu().numpy())
        pass

    scores = np.concatenate(scores_list, axis=0)

    np.save(os.path.join(model_save_path, f'pred_{iepoch:03d}.npy'), scores)
    pass


if __name__ == '__main__':
    main()
    pass