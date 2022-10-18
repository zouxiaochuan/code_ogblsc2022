import datasets6 as datasets
import torch.utils.data
from config2 import config
import ex3_models as models
import torch_utils
import torch.optim
import timm.scheduler
import torch.nn as nn
from tqdm import tqdm
import numpy as np


device = 'cuda:0'

def main():
    model = models.MoleculeHLGapPredictor(config)
    
    dataset_train = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='valid-train', rotate=True, path_atom_map=None, data_path_name='data')
    dataset_test = datasets.SimplePCQM4MDataset(
        path=config['middle_data_path'], split_name='valid-test', rotate=False, path_atom_map=None, data_path_name='data')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn
    )

    # print([n for n, p in model.named_parameters()])
    optimizer = torch.optim.AdamW(torch_utils.get_optimizer_params(model, config['learning_rate'], config['weight_decay']))
    scheduler = timm.scheduler.StepLRScheduler(
        optimizer, decay_t=5, decay_rate=config['learning_rate_decay_rate'],
        warmup_t=config['warmup_epochs'], warmup_lr_init=1e-6)
    
    model.to(device)

    for iepoch in range(config['num_epochs']):
        model.train()
        pbar = tqdm(loader_train)
        running_loss = None
        for ibatch, batch in enumerate(pbar):
            graph, y = batch
            graph = torch_utils.batch_to_device(graph, device)
            y = y.to(device)
            scores = model(
                # graph['atom_feat_cate'],
                # graph['atom_feat_float'], 
                # graph['atom_mask'],
                # graph['bond_index'],
                # graph['bond_feat_cate'],
                # graph['bond_feat_float'],
                # graph['bond_mask'],
                # graph['structure_feat_cate'], 
                # graph['structure_feat_float'],
                # graph['triplet_feat_cate']
                graph
            )
            loss = nn.functional.l1_loss(scores.flatten(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

            if running_loss is None:
                running_loss = loss
            else:
                running_loss = 0.99 * running_loss + 0.01 * loss
                pass

            pbar.set_postfix(loss=running_loss, lr=optimizer.param_groups[0]['lr'])
            pass

        model.eval()

        losses = []
        for batch in tqdm(loader_test):
            graph, y = batch
            graph = torch_utils.batch_to_device(graph, device)
            y = y.to(device)
            with torch.no_grad():
                scores = model(
                    # graph['atom_feat_cate'],
                    # graph['atom_feat_float'], 
                    # graph['atom_mask'],
                    # graph['bond_index'],
                    # graph['bond_feat_cate'],
                    # graph['bond_feat_float'],
                    # graph['bond_mask'],
                    # graph['structure_feat_cate'], 
                    # graph['structure_feat_float'],
                    # graph['triplet_feat_cate']
                    graph
                )
                loss = nn.functional.l1_loss(scores.flatten(), y)
                pass
            losses.append(loss.item())
            pass

        print(f'epoch: {iepoch}, loss: {np.mean(losses)}')
        scheduler.step(iepoch)
        pass
    pass


if __name__ == '__main__':
    main()
    pass