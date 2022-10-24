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
import sys
import torch.cuda
import torch

run_id = 'dropout0.1_decay1_0.97_h32s32_hidden256_classnorm_lesse_trainf0'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    pass



def main(rank, num_processes):
    
    setup(rank, num_processes)

    dataset_train = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='train', rotate=True, data_path_name='data2', load_dist=True)
    dataset_test = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='train_valid_fold0_test', rotate=False,  data_path_name='data2', load_dist=True)

    if rank == 0:
        print(f'num train: {len(dataset_train)}')
        print(f'num test: {len(dataset_test)}')
        pass

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
    model = models.MoleculePairDistClassifier(config)
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
            running_stats = dict()
            pass

        for ibatch, batch in enumerate(loader_train):
            graph, y = batch
            # print(type(graph['num_atom']))
            graph = torch_utils.batch_to_device(graph, device)
            y = y.to(device)
            # print(graph.keys())
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
                graph['triplet_feat_cate']
                # graph
            )

            dist_class = graph['dist_class']
            mask = graph['atom_mask'][:, :, None] * graph['atom_mask'][:, None, :]
            mask = mask * (1 - torch.eye(mask.shape[1], device=mask.device)[None, :, :])
            mask = mask>0
            # print(scores.shape)
            # print(mask.shape)
            scores = scores[mask, :]
            label = dist_class[mask, :]
            # loss = nn.functional.l1_loss(scores, y)
            loss = nn.functional.binary_cross_entropy_with_logits(scores, label)
            accuracy = ((scores>0.5).float() == label).float().mean()
            # accuracy = (scores.argmax(dim=-1) == label).float().mean()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), config['max_grad_norm'])
            optimizer.step()

            loss = loss.item()

            if rank == 0:
                stats = {'loss': loss, 'accuracy': accuracy.item(), 'grad_norm': grad_norm.item()}

                for k, v in stats.items():
                    if k not in running_stats:
                        running_stats[k] = v
                        pass
                    running_stats[k] = 0.99 * running_stats[k] + 0.01 * v
                    pass

                running_stats['lr'] = optimizer.param_groups[0]['lr']
                pbar.set_postfix(**running_stats)
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
                    graph['triplet_feat_cate']
                    # graph
                )
                dist_class = graph['dist_class']
                mask = graph['atom_mask'][:, :, None] * graph['atom_mask'][:, None, :]
                mask = mask * (1 - torch.eye(mask.shape[1], device=mask.device)[None, :, :])
                mask = mask>0
                
                scores = scores[mask, :]
                label = dist_class[mask, :]
                # loss = nn.functional.l1_loss(scores, y)
                loss = nn.functional.binary_cross_entropy_with_logits(scores, label)
                pass

            dist.reduce(loss, 0, op=dist.ReduceOp.SUM)

            if rank == 0:
                losses.append(loss.item() / num_processes)
                pass
            pass
        
        if rank == 0:
            mean_loss = np.mean(losses)
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