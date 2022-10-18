import datasets
import torch.utils.data
from config import config
import ce_models
import torch_utils
import torch.optim
import timm.scheduler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import global_data
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist

run_id = 'cec_dropout_decay5_0.8_h16_fastedge_inter8'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    pass


def get_predicts():
    run_ids = [
        'dropout_decay5_0.8_h16_fastedge_inter8', 
        'ce1_dropout_decay5_0.8_h16_fastedge_inter8', 
        'ce2_dropout_decay5_0.8_h16_fastedge_inter8'
    ]

    iepoch = 40
    scores_list = []
    for run_id in run_ids:
        model_save_path = os.path.join('models_valid', run_id)
        scores = np.load(os.path.join(model_save_path, f'pred_{iepoch:03d}.npy'))
        scores_list.append(scores)
        pass

    scores = np.stack(scores_list, axis=1)
    
    return scores


def main(rank, num_processes):
    
    setup(rank, num_processes)

    proba = np.load(os.path.join(config['middle_data_path'], 'cluster_ensemble', 'proba.npy'))
    predicts = get_predicts()

    extra_data = np.concatenate([proba, predicts], axis=1)
    
    dataset_train = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='train', rotate=True, extra_data=extra_data)
    dataset_test = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='valid', rotate=False, extra_data=extra_data)

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
    model = ce_models.MoleculeClassifier(config, proba.shape[1])

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
            logits = ddp_model(graph)
            gt = graph['extra_data'][:, :3]
        
            loss = torch.mean(
                F.kl_div(F.log_softmax(logits, dim=-1), gt, reduction='batchmean'),
                dim=0)

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

                pbar.set_postfix(loss=format(running_loss, '.4f'), lr=optimizer.param_groups[0]['lr'])
                pbar.update(1)
                pass
            pass

        ddp_model.eval()

        if rank == 0:
            pbar.close()
            losses_reg = []
            losses_cls = []
            pass

        for batch in loader_test:
            graph, y = batch
            graph = torch_utils.batch_to_device(graph, device)
            y = y.to(device)
            with torch.no_grad():
                logits = ddp_model(graph)
                gt = graph['extra_data'][:, :3]
                loss_cls = torch.sum(
                    F.kl_div(F.log_softmax(logits, dim=-1), gt, reduction='sum'),
                    dim=0)
                origin_preds = graph['extra_data'][:, 3:]
                pred_fin = torch.sum(F.softmax(logits, dim=-1) * origin_preds, dim=-1)

                loss_reg = torch.sum(
                    nn.functional.l1_loss(pred_fin, y, reduction='sum'))

                pass

            dist.all_reduce(loss_reg, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_cls, op=dist.ReduceOp.SUM)

            if rank == 0:
                losses_reg.append(loss_reg.item())
                losses_cls.append(loss_cls.item())
                pass
            pass
        
        if rank == 0:
            mean_loss_reg = np.sum(losses_reg) / len(dataset_test)
            mean_loss_cls = np.sum(losses_cls) / len(dataset_test)
            print(f'epoch: {iepoch}, loss_reg: {mean_loss_reg}, loss_cls: {mean_loss_cls}')
            torch.save(
                ddp_model.state_dict(),
                os.path.join(model_save_path, f'epoch_{iepoch:03d}.pt'))
            with open(os.path.join(model_save_path, 'result.txt'), 'a') as f:
                f.write(f'epoch: {iepoch}, loss_reg: {mean_loss_reg}, loss_cls: {mean_loss_cls} \n')
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    mp.spawn(main, nprocs=8, args=(8, ), join=True,)
    pass