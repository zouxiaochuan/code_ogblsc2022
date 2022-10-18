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

            g['structure_feat_float'] = np.zeros((num_atom, num_atom, 1), dtype='float32')
            # g['atom_feat_float'] = np.concatenate((g['atom_feat_float'], g['xyz']), axis=-1)

            if self.extra_data is not None:
                g['extra_data'] = self.extra_data[data_idx].astype('float32')
                pass

            valence = g['atom_feat_cate'][:, 9] + g['atom_feat_cate'][:, 10]
            total_valence = np.sum(valence)
            num_nodes = num_atom + total_valence
            emask = np.zeros(num_nodes, dtype='float32')
            emask[num_atom:] = 1

            nuclei_index = np.arange(num_atom)
            index_node2atom = np.concatenate((nuclei_index, np.repeat(nuclei_index, valence)))
            index_node2atom_2d = np.stack(
                [
                    index_node2atom[:, None].repeat(num_nodes, axis=1),
                    index_node2atom[None, :].repeat(num_nodes, axis=0)
                ],
                axis=2
            )

            node_feat_cate = np.concatenate(
                (
                    g['atom_feat_cate']+1, 
                    np.zeros((total_valence, g['atom_feat_cate'].shape[1]), dtype='int64'),
                ), 
                axis=0
            )
            electron_seq_num = np.zeros(num_nodes, dtype='int64')
            atom_seq = np.zeros(num_nodes, dtype='int64')
            atom_seq = np.zeros(num_atom, dtype='int64')
            for i in range(num_nodes):
                electron_seq_num[i] = atom_seq[index_node2atom[i]]
                atom_seq[index_node2atom[i]] += 1
                pass
            
            node_feat_cate = np.concatenate((node_feat_cate, electron_seq_num[:, None]), axis=1)
            node_feat_float = np.concatenate(
                (
                    g['atom_feat_float'],
                    np.zeros((total_valence, g['atom_feat_float'].shape[1]), dtype='float32')
                ),
                axis=0
            )

            bond_n2n = g['bond_index']
            bond_feat_cate_map = np.zeros(
                (num_atom, num_atom, g['bond_feat_cate'].shape[1]), dtype='int64'
            )
            bond_feat_cate_map[bond_n2n[0], bond_n2n[1]] = g['bond_feat_cate'] + 1
            # edge_feat_bond_cate = bond_feat_cate_map[index_node2atom_2d[:, :, 0], index_node2atom_2d[:, :, 1]]
            edge_feat_bond_cate = np.zeros(
                (num_nodes, num_nodes, g['bond_feat_cate'].shape[1]), dtype='int64'
            )
            edge_feat_bond_cate[bond_n2n[0], bond_n2n[1]] = g['bond_feat_cate'] + 1
            edge_feat_is_same_atom = (index_node2atom_2d[:, :, 0] == index_node2atom_2d[:, :, 1]).astype('int64')
            edge_feat_structure_cate = np.zeros(
                (num_nodes, num_nodes, g['structure_feat_cate'].shape[2]), dtype='int64'
            )
            edge_feat_structure_cate[:num_atom, :num_atom, :] = g['structure_feat_cate'] + 1

            edge_feat_type = np.zeros(
                (num_nodes, num_nodes), dtype='int64'
            )
            # e2e
            edge_feat_type[
                (index_node2atom_2d[:, :, 0] >= num_atom) & (index_node2atom_2d[:, :, 1] >= num_atom)] = 0
            # e2n
            edge_feat_type[
                (index_node2atom_2d[:, :, 0] >= num_atom) & (index_node2atom_2d[:, :, 1] < num_atom)] = 1
            # n2e
            edge_feat_type[
                (index_node2atom_2d[:, :, 0] < num_atom) & (index_node2atom_2d[:, :, 1] >= num_atom)] = 2
            # n2n
            edge_feat_type[
                (index_node2atom_2d[:, :, 0] < num_atom) & (index_node2atom_2d[:, :, 1] < num_atom)] = 3

            edge_feat_bond_float = np.zeros(
                (num_nodes, num_nodes, g['bond_feat_float'].shape[1]), dtype='float32'
            )
            bond_feat_float_map = np.zeros(
                (num_atom, num_atom, g['bond_feat_float'].shape[1]), dtype='float32'
            )
            bond_feat_float_map[bond_n2n[0], bond_n2n[1]] = g['bond_feat_float']
            edge_feat_bond_float = bond_feat_float_map[index_node2atom_2d[:, :, 0], index_node2atom_2d[:, :, 1]]
            edge_feat_structure_float = np.zeros(
                (num_nodes, num_nodes, g['structure_feat_float'].shape[2]), dtype='float32')
            edge_feat_structure_float[:num_atom, :num_atom, :] = g['structure_feat_float']


            g['node_feat_cate'] = node_feat_cate
            g['node_feat_float'] = node_feat_float
            g['structure_feat_cate'] = np.concatenate(
                (edge_feat_bond_cate, edge_feat_is_same_atom[:, :, None], edge_feat_structure_cate,
                 edge_feat_type[:, :, None]), axis=-1
            )
            g['structure_feat_float'] = np.concatenate(
                (edge_feat_bond_float, edge_feat_structure_float), axis=-1
            )
            g['num_atom'] = num_atom

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
    xyz = common_utils.collate_seq([g['xyz'] for g in graph_list])[0]
    atom_feat_cate, atom_mask = common_utils.collate_seq([g['atom_feat_cate'] for g in graph_list])

    structure_feat_cate = common_utils.collate_map([g['structure_feat_cate'] for g in graph_list])
    structure_feat_float = common_utils.collate_map([g['structure_feat_float'] for g in graph_list])
    num_atom = np.max([g['num_atom'] for g in graph_list])
    
    result_dict = {
        'node_feat_cate': torch.from_numpy(node_feat_cate),
        'node_feat_float': torch.from_numpy(node_feat_float),
        'node_mask': torch.from_numpy(node_mask),
        'structure_feat_cate': torch.from_numpy(structure_feat_cate),
        'structure_feat_float': torch.from_numpy(structure_feat_float),
        'xyz': torch.from_numpy(xyz),
        'atom_mask': torch.from_numpy(atom_mask),
        'num_atom': num_atom
    }

    if 'extra_data' in graph_list[0]:
        result_dict['extra_data'] = torch.from_numpy(
            np.stack([g['extra_data'] for g in graph_list]))
        pass

    return result_dict, torch.tensor(y, dtype=torch.float32)