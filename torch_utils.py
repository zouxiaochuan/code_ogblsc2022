import torch

def get_optimizer_params(
        model: torch.nn.Module, learning_rate, weight_decay, no_decay_keys=['bias', 'layer_norm']):
    
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay_keys)],
            'lr': learning_rate, 'weight_decay': 0.0 },
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay_keys)],
            'lr': learning_rate, 'weight_decay': weight_decay }
    ]
    return optimizer_parameters


def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
        pass
    elif isinstance(batch, dict):
        b = dict()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                b[k] = batch_to_device(v, device)
                pass
            else:
                b[k] = v
            pass
        return b
        pass
    else:
        raise RuntimeError('type not supported')
    pass


def dist_loss(pred, gt, mask, rate=False, thr=0, p=1):
    mask_ = mask[:, :, None] * mask[:, None, :]
    num = torch.sum(mask_)

    if p == 2:
        diff = (pred - gt) ** 2
        pass
    else:
        diff = torch.abs(pred - gt)
        pass
    if rate:
        diff = diff / (gt + 1e-12)
        pass
    mask_ = mask_ * (diff > thr)
    num = torch.sum(mask_)
    loss = torch.sum(diff * mask_) / (num+1e-12)
    return loss

