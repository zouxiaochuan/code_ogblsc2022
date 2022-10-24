import features
import os


config = {
    'num_epochs': 100,
    'batch_size': 64,
    'hidden_size': 256,
    'head_size': 32,
    'num_heads': 32,
    'intermediate_size': 256 * 8,
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'input_dropout_prob': 0.1,
    'num_data_workers': 4,
    'nums_atom_feat_cate': features.ATOM_CATE_FEAT_DIMS,
    # 'nums_atom_feat_cate': [2],
    'num_atom_feat_float': features.ATOM_FLOAT_FEAT_DIM ,
    # 'num_atom_feat_float': 5,
    'nums_bond_feat_cate': features.BOND_CATE_FEAT_DIMS,
    'num_bond_feat_float': features.BOND_FLOAT_FEAT_DIM,
    'num_spread_layers': 24,
    'nums_structure_feat_cate': [
        features.MAX_SHORTEST_PATH_LEN + 1, features.MAX_SAME_RING_COUNT, features.MAX_SAME_RING_MIN_SIZE,
        # features.MAX_ANGLES
    ],
    # 'nums_structure_feat_cate': [2],
    'num_structure_feat_float': 16,
    # 'num_structure_feat_float': 3,
    'nums_triplet_feat_cate': [
        features.MAX_SHORTEST_PATH_LEN + 1, features.MAX_SHORTEST_PATH_LEN + 1, 2],
    'learning_rate_decay_rate': 0.97,
    'learning_rate': 1e-4,
    'learning_rate_min': 1e-6,
    'warmup_epochs': 5,
    'weight_decay': 1e-8,
    'middle_data_path': os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m'),
    'ogb_data_path': '../ogblsc_data',
    'num_cluster': 3,
    'temperature': 0.1,
    'num_dist_class': 16,
    'max_grad_norm': 100
}
