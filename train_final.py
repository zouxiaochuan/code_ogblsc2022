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
import global_data
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist
import sys


train_split = 'train_valid_fold49'
test_split = 'train_valid_fold49_test'
run_id = 'final49_dropout_decay1_0.97_h16s16_hidden256_fastedge_l24'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    pass

def main(rank, num_processes):
    
    setup(rank, num_processes)

    dataset_train = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name=train_split, rotate=True, path_atom_map=None, data_path_name='data')
    dataset_test = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name=test_split, rotate=False, path_atom_map=None, data_path_name='data')

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
    model = models.MoleculeHLGapPredictor(config)
    if rank == 0:
        print('num of parameters: {0}'.format(np.sum([p.numel() for p in model.parameters()])))
        pass
    
    model.to(device)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    if len(sys.argv) > 1:
        ddp_model.load_state_dict(torch.load(sys.argv[1]))
        pass

    optimizer = torch.optim.AdamW(torch_utils.get_optimizer_params(ddp_model, config['learning_rate'], config['weight_decay']))
    scheduler = timm.scheduler.StepLRScheduler(
        optimizer, decay_t=1, decay_rate=config['learning_rate_decay_rate'],
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
                graph['structure_feat_float'],
                graph['triplet_feat_cate'])[0]
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
                    graph['structure_feat_float'],
                    graph['triplet_feat_cate'])[0]
                loss = nn.functional.l1_loss(scores, y, reduction='sum')
                pass
            dist.reduce(loss, 0, op=dist.ReduceOp.SUM)

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
    num_gpus = torch.cuda.device_count()
    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, ), join=True,)
    pass