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

run_id = 'ced_dropout_decay5_0.8_h16_fastedge_inter8'

device = 'cuda:0'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    pass

def main():
    
    dataset_train = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='train', rotate=True)
    dataset_test = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='valid', rotate=False)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config['batch_size'],
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn,
        shuffle=True
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config['batch_size'],
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn,
        shuffle=False
    )

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    model = ce_models.ClusterEnsemble(config)
    print('num of parameters: {0}'.format(np.sum([p.numel() for p in model.parameters()])))
    print('num of training data: {0}'.format(len(dataset_train)))
    print('num of testing data: {0}'.format(len(dataset_test)))
    
    model.to(device)
    optimizer = torch.optim.AdamW(torch_utils.get_optimizer_params(model, config['learning_rate'], config['weight_decay']))
    scheduler = timm.scheduler.StepLRScheduler(
        optimizer, decay_t=5, decay_rate=config['learning_rate_decay_rate'],
        warmup_t=config['warmup_epochs'], warmup_lr_init=1e-6)

    model_save_path = os.path.join('models_valid', run_id)

    if os.path.exists(model_save_path):
        raise RuntimeError('model_save_path already exists')
        pass

    os.makedirs(model_save_path, exist_ok=True)
    pass

    for iepoch in range(config['num_epochs']):
        scheduler.step(iepoch)
        model.train()

        pbar = tqdm(loader_train)
        running_loss = None

        for ibatch, batch in enumerate(loader_train):
            graph, y = batch
            graph = torch_utils.batch_to_device(graph, device)
            y = y.to(device)
            preds, logits = model(graph)
            
            if ibatch <= 1000:
                temp = 100
            else:
                temp = config['temperature']
                pass
            loss_reg, loss_cls, loss_fin = ce_models.loss_cluster_ensemble(preds, logits, y, temp)
            loss = loss_reg + loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = np.array([loss_reg.item(), loss_cls.item(), loss_fin.item()])


            if running_loss is None:
                running_loss = loss
            else:
                running_loss = 0.99 * running_loss + 0.01 * loss
                pass

            postfix = {'lr': f'{optimizer.param_groups[0]["lr"]:0.4f}'}
            for i, l in enumerate(running_loss):
                postfix[f'loss{i}'] = f'{l:0.4f}'              
                pass
            pbar.set_postfix(postfix)
            pbar.update(1)
            pass

        model.eval()

        pbar.close()
        losses = []

        for batch in loader_test:
            graph, y = batch
            graph = torch_utils.batch_to_device(graph, device)
            y = y.to(device)
            with torch.no_grad():
                preds, logits = model(graph)
                _, _, loss_fin = ce_models.loss_cluster_ensemble(preds, logits, y, config['temperature'])
                pass

            losses.append(loss_fin.item())
            pass
        
        mean_loss = np.mean(losses)
        print(f'epoch: {iepoch}, loss: {mean_loss}')
        torch.save(
            model.state_dict(),
            os.path.join(model_save_path, f'epoch_{iepoch:03d}.pt'))
        with open(os.path.join(model_save_path, 'result.txt'), 'a') as f:
            f.write(f'epoch: {iepoch}, loss: {mean_loss}\n')
            pass
        pass
    pass


if __name__ == '__main__':
    main()
    pass