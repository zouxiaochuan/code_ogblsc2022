import datasets
import torch.utils.data
from config import config
import ce_models
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
import cluster_utils

run_id = 'boost02_cls0.02_dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles'
iepoch = 30
device = 'cuda:1'



def main():

    y_clusters = cluster_utils.load_clusters(config['middle_data_path'], '02')
    dataset = datasets.SimplePCQM4MDataset(path=config['middle_data_path'], split_name='all', rotate=False)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn,
        shuffle=False
    )

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    model = ce_models.MoleculeClassifier(config, y_clusters.shape[0])
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
    y_clusters = torch.from_numpy(y_clusters).to(device).float()
    for batch in tqdm(loader):
        graph, y = batch
        graph = torch_utils.batch_to_device(graph, device)
        with torch.no_grad():
            logits = model(graph)
            proba = torch.softmax(logits, dim=1)
            # pred = y_clusters[:, 0][torch.argmax(proba, dim=1)]
            pred = torch.sum(proba * y_clusters[:, 0][None, :], dim=1)
            pass
            scores_list.append(pred.detach().cpu().numpy())
        pass

    scores = np.concatenate(scores_list, axis=0)

    np.save(os.path.join(model_save_path, f'pred_{iepoch:03d}.npy'), scores)
    pass


if __name__ == '__main__':
    main()
    pass