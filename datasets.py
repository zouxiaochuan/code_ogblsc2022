import torch
import common_utils
import os
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import global_data


class SimplePCQM4MDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, split_name='train', rotate=False, subset=None, extra_data=None, path_atom_map=None, data_path_name='data2'):
        self.path = path
        self.idx_split = common_utils.load_obj(os.path.join(path, 'idx_split.pkl'))
        train_split = self.idx_split['train']
        self.idx_split['valid-train'] = train_split[: len(train_split) * 9 // 10]
        self.idx_split['valid-test'] = train_split[len(train_split) * 9 // 10:]
        self.idx_split['all'] = np.arange(3746620)

        self.idx_split['train_valid'] = np.concatenate(
            [self.idx_split['train'], self.idx_split['valid']])
        
        for i in range(50):
            self.idx_split[f'train_valid_fold{i}'] = self.idx_split['train_valid'][
                np.arange(len(self.idx_split['train_valid'])) % 50 != i]
            self.idx_split[f'train_valid_fold{i}_test'] = self.idx_split['train_valid'][
                np.arange(len(self.idx_split['train_valid'])) % 50 == i]
            pass
        
        self.data_path = os.path.join(path, data_path_name)

        self.split = self.idx_split[split_name]
        self.rotate = rotate
        # if split_name == 'train':
        #     self.split = self.split[:5012]
        if subset is not None:
            subset = subset[self.split]
            self.split = self.split[subset]
            pass
        self.extra_data = extra_data
        if path_atom_map is not None:
            self.path_atom_map = common_utils.load_obj(os.path.join(path, path_atom_map))
            pass
        else:
            self.path_atom_map = None
            pass
        pass
        
    
    def get_data(self, data_idx):
        filename = os.path.join(
            self.data_path, format(data_idx // 1000, '04d'), format(data_idx, '07d') + '.pkl')

        with open(filename, 'rb') as fin:
            g, y = pickle.load(fin)
            pass

        return g, y

        # return global_data.DATA[data_idx]
        pass

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, (int, np.integer)):
            data_idx = self.split[idx]

            g, y = self.get_data(data_idx)

            if self.rotate:
                mat3d = R.random().as_matrix().astype('float32')
                g['xyz'] = np.matmul(g['xyz'], mat3d)
                pass
            
            num_atom = g['xyz'].shape[0]


            g['structure_feat_cate'] = np.stack(
                [
                    g['shortest_path_length'],
                    g['atom_same_ring_count'],
                    g['atom_same_ring_min_size'],
                    # g['angles'],
                ], axis=2
            ).astype('int64')

            # ijk = ik + jk
            # triplet_shortest_path_length = g['shortest_path_length'][:, None, :] + g['shortest_path_length'][None, :, :]
            # triplet_shortest_path_min = np.minimum(g['shortest_path_length'][:, None, :], g['shortest_path_length'][None, :, :])
            # triplet_shortest_path_max = np.maximum(g['shortest_path_length'][:, None, :], g['shortest_path_length'][None, :, :])

            # ijk = ik
            triplet_shortest_path_length_src = np.tile(g['shortest_path_length'][:, None, :], (1, num_atom, 1))
            # ijk = jk
            triplet_shortest_path_length_dst = np.tile(g['shortest_path_length'][None, :, :], (num_atom, 1, 1))

            shortest_path = g['shortest_path']
            path_idx = np.array([(i, j, k) for i in range(num_atom) for j in range(num_atom) for k in shortest_path[i][j]])

            triplet_in_shortest_path = np.zeros((num_atom, num_atom, num_atom), dtype='int64')

            if len(path_idx) > 0:
                triplet_in_shortest_path[path_idx[:, 0], path_idx[:, 1], path_idx[:, 2]] = 1
                pass

            g['triplet_feat_cate'] = np.stack(
                [triplet_shortest_path_length_src, triplet_shortest_path_length_dst, triplet_in_shortest_path],
                axis=3
            ).astype('int64')

            # print(g['xyz'])
            shift = g['xyz'][None, :, :] - g['xyz'][:, None, :]
            dist = np.linalg.norm(shift, axis=-1, keepdims=True)
            angle = np.divide(shift, dist+1e-12)
            mid = (g['xyz'][None, :, :] + g['xyz'][:, None, :]) * 0.5
            # g['structure_feat_float'] = dist
            g['structure_feat_float'] = np.zeros((num_atom, num_atom, 1), dtype='float32')
            # g['atom_feat_float'] = np.concatenate((g['atom_feat_float'], g['xyz']), axis=-1)

            if self.extra_data is not None:
                g['extra_data'] = self.extra_data[data_idx].astype('float32')
                pass

            return g, y

        raise IndexError(
            'Only integer is valid index (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.split)

    pass


def collate_fn(graph_list):
    y = [g[1] for g in graph_list]
    graph_list = [g[0] for g in graph_list]
    atom_feat_cate, atom_mask = common_utils.collate_seq([g['atom_feat_cate'] for g in graph_list])
    bond_feat_cate, bond_mask = common_utils.collate_seq([g['bond_feat_cate'] for g in graph_list])
    xyz = common_utils.collate_seq([g['xyz'] for g in graph_list])[0]

    atom_feat_float, _ = common_utils.collate_seq([g['atom_feat_float'] for g in graph_list])
    bond_feat_float, _ = common_utils.collate_seq([g['bond_feat_float'] for g in graph_list])

    bond_index, _ = common_utils.collate_seq([g['bond_index'].T for g in graph_list])
    bond_index = bond_index.transpose(0, 2, 1).astype('int64')

    structure_feat_cate = common_utils.collate_map([g['structure_feat_cate'] for g in graph_list])
    structure_feat_float = common_utils.collate_map([g['structure_feat_float'] for g in graph_list])
    triplet_feat_cate = common_utils.collate_cube([g['triplet_feat_cate'] for g in graph_list])
    
    result_dict = {
        'atom_feat_cate': torch.from_numpy(atom_feat_cate),
        'bond_feat_cate': torch.from_numpy(bond_feat_cate),
        'atom_feat_float': torch.from_numpy(atom_feat_float),
        'bond_feat_float': torch.from_numpy(bond_feat_float),
        'bond_index': torch.from_numpy(bond_index),
        'atom_mask': torch.from_numpy(atom_mask),
        'bond_mask': torch.from_numpy(bond_mask),
        'structure_feat_cate': torch.from_numpy(structure_feat_cate),
        'structure_feat_float': torch.from_numpy(structure_feat_float),
        'triplet_feat_cate': torch.from_numpy(triplet_feat_cate),
        'xyz': torch.from_numpy(xyz)
    }

    if 'extra_data' in graph_list[0]:
        result_dict['extra_data'] = torch.from_numpy(
            np.stack([g['extra_data'] for g in graph_list]))
        pass

    return result_dict, torch.tensor(y, dtype=torch.float32)