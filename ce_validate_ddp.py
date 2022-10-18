import datasets
import torch.utils.data
from config import config
import models
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

run_id = 'ceo1_dropout_decay5_0.8_h16_fastedge_inter8_2'
subset_filenames = [
    'idx_0.npy', 
    # 'idx_rest_1.npy', 'idx_rest_2.npy'
]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    pass

def main(rank, num_processes):
    
    setup(rank, num_processes)

    subset = np.load(os.path.join(config['middle_data_path'], 'cluster_ensemble', subset_filenames[0]))

    for subset_filename in subset_filenames[1:]:
        subset = subset & np.load(os.path.join(config['middle_data_path'], 'cluster_ensemble', subset_filename))
        pass
    
    dataset_train = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='train', rotate=True, subset=subset)
    dataset_test = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='valid', rotate=False, subset=subset)

    sampler_train = DistributedSampler(dataset_train, shuffle=True)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config['batch_size'],
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn,
        sampler=sampler_train
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config['batch_size'],
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn,
        sampler=DistributedSampler(dataset_test, shuffle=False)
    )

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    device = f'cuda:{rank}'
    model = models.MoleculeNet(config)
    if rank == 0:
        print('num of parameters: {0}'.format(np.sum([p.numel() for p in model.parameters()])))
        print('num of training data: {0}'.format(len(dataset_train)))
        print('num of testing data: {0}'.format(len(dataset_test)))
        pass
    
    model.to(device)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(torch_utils.get_optimizer_params(ddp_model, config['learning_rate'], config['weight_decay']))
    scheduler = timm.scheduler.StepLRScheduler(
        optimizer, decay_t=5, decay_rate=config['learning_rate_decay_rate'],
        warmup_t=config['warmup_epochs'], warmup_lr_init=1e-6)

    model_save_path = os.path.join('models_valid', run_id)

    if rank == 0:
        if os.path.exists(model_save_path):
            raise RuntimeError('model_save_path already exists')
            pass

        os.makedirs(model_save_path, exist_ok=True)
        pass

    for iepoch in range(config['num_epochs']):
        sampler_train.set_epoch(iepoch)
        scheduler.step(iepoch)
        ddp_model.train()

        if rank == 0:
            pbar = tqdm(loader_train)
            running_loss = None
            pass

        for ibatch, batch in enumerate(loader_train):
            graph, y = batch
            graph = torch_utils.batch_to_device(graph, device)
            y = y.to(device)
            scores = ddp_model(
                graph['atom_feat_cate'],
                graph['atom_feat_float'], 
                graph['atom_mask'],
                graph['bond_index'], 
                graph['bond_feat_cate'], 
                graph['bond_feat_float'],
                graph['bond_mask'],
                graph['structure_feat_cate'], 
                graph['structure_feat_float'])
            loss = nn.functional.l1_loss(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

            if rank == 0:
                if running_loss is None:
                    running_loss = loss
                else:
                    running_loss = 0.99 * running_loss + 0.01 * loss
                    pass

                pbar.set_postfix(loss=running_loss, lr=optimizer.param_groups[0]['lr'])
                pbar.update(1)
                pass
            pass

        ddp_model.eval()

        if rank == 0:
            pbar.close()
            losses = []
            pass

        for batch in loader_test:
            graph, y = batch
            graph = torch_utils.batch_to_device(graph, device)
            y = y.to(device)
            with torch.no_grad():
                scores = ddp_model(
                    graph['atom_feat_cate'],
                    graph['atom_feat_float'], 
                    graph['atom_mask'],
                    graph['bond_index'], 
                    graph['bond_feat_cate'], 
                    graph['bond_feat_float'],
                    graph['bond_mask'],
                    graph['structure_feat_cate'], 
                    graph['structure_feat_float'])
                loss = nn.functional.l1_loss(scores, y, reduction='sum')
                pass
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            if rank == 0:
                losses.append(loss.item())
                pass
            pass
        
        if rank == 0:
            mean_loss = np.sum(losses) / len(dataset_test)
            print(f'epoch: {iepoch}, loss: {mean_loss}')
            torch.save(
                ddp_model.state_dict(),
                os.path.join(model_save_path, f'epoch_{iepoch:03d}.pt'))
            with open(os.path.join(model_save_path, 'result.txt'), 'a') as f:
                f.write(f'epoch: {iepoch}, loss: {mean_loss}\n')
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    mp.spawn(main, nprocs=8, args=(8, ), join=True,)
    pass