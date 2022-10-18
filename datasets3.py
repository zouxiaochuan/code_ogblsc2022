import torch
import common_utils
import os
import numpy as np
import pickle
import random


class SimplePCQM4MDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, split_name='train', rotate=False, subset=None, extra_data=None, path_atom_map=None):
        self.path = path
        self.idx_split = common_utils.load_obj(os.path.join(path, 'idx_split.pkl'))
        train_split = self.idx_split['train']
        self.idx_split['valid-train'] = train_split[: len(train_split) * 7 // 10]
        self.idx_split['valid-test'] = train_split[len(train_split) * 7 // 10:]
        self.idx_split['all'] = np.arange(3746620)

        self.idx_split['train_valid'] = np.concatenate(
            [self.idx_split['train'], self.idx_split['valid']])
        
        for i in range(50):
            self.idx_split[f'train_valid_fold{i}'] = self.idx_split['train_valid'][
                np.arange(len(self.idx_split['train_valid'])) % 50 != i]
            self.idx_split[f'train_valid_fold{i}_test'] = self.idx_split['train_valid'][
                np.arange(len(self.idx_split['train_valid'])) % 50 == i]
            pass
        
        self.data_path = os.path.join(path, 'data')

        self.split = self.idx_split[split_name]
        self.split_name = split_name
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

            num_atom = g['atom_feat_cate'].shape[0]
            num_bond = g['bond_feat_cate'].shape[0]

            if self.split_name == 'train' and random.random() < 0.1:
                aidx = random.randint(0, num_atom-1)
                g['atom_feat_cate'][aidx] = 0
                g['atom_feat_float'][aidx] = 0
                pass

            if num_bond == 0 and num_atom == 1:
                bond_index = np.zeros((2, 1), dtype=np.int64)
                bond_feat_cate = np.zeros((1, g['bond_feat_cate'].shape[1]), dtype=np.int64)
                bond_feat_float = np.zeros((1, g['bond_feat_float'].shape[1]), dtype=np.float32)
                num_bond = 1
            elif num_bond == 0 and num_atom > 1:
                num_bond = num_atom
                bond_index = np.zeros((2, num_bond), dtype=np.int64)
                # generate combinations of atom indices
                for i in range(num_atom):
                    bond_index[0, i] = i
                    bond_index[1, i] = (i+1) % num_atom
                    pass

                bond_feat_cate = np.zeros((num_bond, g['bond_feat_cate'].shape[1]), dtype=np.int64)
                bond_feat_float = np.zeros((num_bond, g['bond_feat_float'].shape[1]), dtype=np.float32)
            else:
                bond_index = g['bond_index']
                bond_feat_cate = g['bond_feat_cate']
                bond_feat_float = g['bond_feat_float']
                pass

            node_feat_cate_src = g['atom_feat_cate'][bond_index[0]]
            node_feat_cate_dst = g['atom_feat_cate'][bond_index[1]]
            node_feat_cate = np.concatenate(
                [node_feat_cate_src, node_feat_cate_dst, bond_feat_cate], axis=1)
            
            node_feat_float_src = g['atom_feat_float'][bond_index[0]]
            node_feat_float_dst = g['atom_feat_float'][bond_index[1]]
            node_feat_float = np.concatenate(
                [node_feat_float_src, node_feat_float_dst, bond_feat_float], axis=1)
            
            g['node_feat_cate'] = node_feat_cate
            g['node_feat_float'] = node_feat_float

            from_src_atom = np.tile(bond_index[0, :, None], [1, num_bond])
            to_src_atom = np.tile(bond_index[None, 0, :], [num_bond, 1])
            from_dst_atom = np.tile(bond_index[1, :, None], [1, num_bond])
            to_dst_atom = np.tile(bond_index[None, 1, :], [num_bond, 1])

            edge_src2src_atom = np.stack(
                [from_src_atom, to_src_atom], axis=-1
            )
            edge_src2dst_atom = np.stack(
                [from_src_atom, to_dst_atom], axis=-1
            )
            edge_dst2src_atom = np.stack(
                [from_dst_atom, to_src_atom], axis=-1
            )
            edge_dst2dst_atom = np.stack(
                [from_dst_atom, to_dst_atom], axis=-1
            )

            edge_src2src_mask = edge_src2src_atom[:, :, 0] == edge_src2src_atom[:, :, 1]
            edge_src2dst_mask = edge_src2dst_atom[:, :, 0] == edge_src2dst_atom[:, :, 1]
            edge_dst2src_mask = edge_dst2src_atom[:, :, 0] == edge_dst2src_atom[:, :, 1]
            edge_dst2dst_mask = edge_dst2dst_atom[:, :, 0] == edge_dst2dst_atom[:, :, 1]

            edge_adj_atom = np.stack(
                [
                    edge_src2src_atom[:, :, 0], edge_src2dst_atom[:, :, 0],
                    edge_dst2src_atom[:, :, 0], edge_dst2dst_atom[:, :, 0]
                ],
                axis=-1)

            edge_adj_mask = np.stack(
                [
                    edge_src2src_mask, edge_src2dst_mask, edge_dst2src_mask, 
                    edge_dst2dst_mask
                ],
                axis=-1)
            
            edge_adj_feat_cate = g['atom_feat_cate'][edge_adj_atom]
            edge_adj_feat_float = g['atom_feat_float'][edge_adj_atom]

            edge_adj_feat_cate += 1
            edge_adj_feat_cate[np.logical_not(edge_adj_mask)] = 0
            edge_adj_feat_float[np.logical_not(edge_adj_mask)] = 0

            edge_adj_feat_cate = edge_adj_feat_cate.reshape(num_bond, num_bond, -1)
            edge_adj_feat_float = edge_adj_feat_float.reshape(num_bond, num_bond, -1)

            shortest_path_src2src = g['shortest_path_length'][edge_src2src_atom[:, :, 0], edge_src2src_atom[:, :, 1]]
            shortest_path_src2dst = g['shortest_path_length'][edge_src2dst_atom[:, :, 0], edge_src2dst_atom[:, :, 1]]
            shortest_path_dst2src = g['shortest_path_length'][edge_dst2src_atom[:, :, 0], edge_dst2src_atom[:, :, 1]]
            shortest_path_dst2dst = g['shortest_path_length'][edge_dst2dst_atom[:, :, 0], edge_dst2dst_atom[:, :, 1]]

            shortest_path_all = np.stack(
                [
                    shortest_path_src2src, shortest_path_src2dst, shortest_path_dst2src,
                    shortest_path_dst2dst
                ],
                axis=-1
            )
            shortest_path_min = np.min(shortest_path_all, axis=-1)
            shortest_path_max = np.max(shortest_path_all, axis=-1)

            edge_same_ring_count = g['atom_same_ring_count'][edge_src2src_atom[:, :, 0], edge_src2src_atom[:, :, 1]]
            edge_same_ring_min_size = g['atom_same_ring_min_size'][edge_src2src_atom[:, :, 0], edge_src2src_atom[:, :, 1]]

            g['structure_feat_cate'] = np.concatenate(
                [
                    edge_adj_feat_cate,
                    shortest_path_all,
                    shortest_path_min[:, :, None],
                    shortest_path_max[:, :, None],
                    edge_same_ring_count[:, :, None],
                    edge_same_ring_min_size[:, :, None]
                ], axis=-1
            ).astype('int64')

            g['structure_feat_float'] = edge_adj_feat_float
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
    node_feat_cate, node_mask = common_utils.collate_seq([g['node_feat_cate'] for g in graph_list])

    node_feat_float, _ = common_utils.collate_seq([g['node_feat_float'] for g in graph_list])

    structure_feat_cate = common_utils.collate_map([g['structure_feat_cate'] for g in graph_list])
    structure_feat_float = common_utils.collate_map([g['structure_feat_float'] for g in graph_list])
    
    result_dict = {
        'node_feat_cate': torch.from_numpy(node_feat_cate),
        'node_feat_float': torch.from_numpy(node_feat_float),
        'node_mask': torch.from_numpy(node_mask),
        'structure_feat_cate': torch.from_numpy(structure_feat_cate),
        'structure_feat_float': torch.from_numpy(structure_feat_float),
    }

    if 'extra_data' in graph_list[0]:
        result_dict['extra_data'] = torch.from_numpy(
            np.stack([g['extra_data'] for g in graph_list]))
        pass

    return result_dict, torch.tensor(y, dtype=torch.float32)