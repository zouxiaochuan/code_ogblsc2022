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

run_id = 'cls0.02_dropout_decay5_0.8_h16_fastedge_inter8_bondreverse_angles'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    pass

def main(rank, num_processes):
    
    setup(rank, num_processes)

    dataset_train = datasets.SimplePCQM4MDataset(path=config['middle_data_path'], split_name='train', rotate=True)
    dataset_test = datasets.SimplePCQM4MDataset(path=config['middle_data_path'], split_name='valid', rotate=False)
    y_clusters = cluster_utils.load_clusters(config['middle_data_path'])

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
    model = ce_models.MoleculeClassifier(config, y_clusters.shape[0])
    if rank == 0:
        print('num of parameters: {0}'.format(np.sum([p.numel() for p in model.parameters()])))
        print('num of training data: {0}'.format(len(dataset_train)))
        print('num of testing data: {0}'.format(len(dataset_test)))
        print('num of clusters: {0}'.format(y_clusters.shape[0]))
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

    y_clusters = torch.from_numpy(y_clusters).to(device).float()
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
            logits = ddp_model(graph)
            y_cls = cluster_utils.get_nearest_cluster(y, y_clusters)
            loss_cls = nn.functional.cross_entropy(logits, y_cls)
            proba = torch.softmax(logits, dim=1)
            pred = torch.sum(proba * y_clusters[:, 0][None, :], dim=1)
            loss_reg = torch.mean(torch.abs(pred - y))

            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()

            loss = loss_reg.item()

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
                logits = ddp_model(graph)
                # y_cls = cluster_utils.get_nearest_cluster(y, y_clusters)
                # loss_cls = nn.functional.cross_entropy(logits, y_cls)
                proba = torch.softmax(logits, dim=1)
                pred = torch.sum(proba * y_clusters[:, 0][None, :], dim=1)
                loss_reg = torch.sum(torch.abs(pred - y))
                loss = loss_reg
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