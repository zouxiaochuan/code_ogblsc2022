import pickle
import numpy as np


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    pass

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    pass


def collate_seq(feat_list):
    batch_size = len(feat_list)
    feat_max_len = np.max([feat.shape[0] for feat in feat_list])
    feat_dim = feat_list[0].shape[1]
    feat = np.zeros(
        (batch_size, feat_max_len, feat_dim),
        dtype=feat_list[0].dtype)
    mask = np.zeros((batch_size, feat_max_len), dtype='float32')

    for i, ifeat in enumerate(feat_list):
        size = ifeat.shape[0]
        feat[i, :size, :] = ifeat
        mask[i, :size] = 1
        pass

    return feat, mask


def collate_map(feat_list):
    batch_size = len(feat_list)
    feat_max_len = np.max([feat.shape[0] for feat in feat_list])
    feat_dim = feat_list[0].shape[2]
    feat = np.zeros(
        (batch_size, feat_max_len, feat_max_len, feat_dim),
        dtype=feat_list[0].dtype)

    for i, ifeat in enumerate(feat_list):
        size = ifeat.shape[0]
        feat[i, :size, :size, :] = ifeat
        pass

    return feat    
    pass

def collate_cube(feat_list):
    batch_size = len(feat_list)
    feat_max_len = np.max([feat.shape[0] for feat in feat_list])
    feat_dim = feat_list[0].shape[3]
    feat = np.zeros(
        (batch_size, feat_max_len, feat_max_len, feat_max_len, feat_dim),
        dtype=feat_list[0].dtype)

    for i, ifeat in enumerate(feat_list):
        size = ifeat.shape[0]
        feat[i, :size, :size, :size, :] = ifeat
        pass

    return feat    
    pass