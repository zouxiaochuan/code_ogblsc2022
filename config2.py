import features
import os

import features

config = {
    'num_epochs': 100,
    'batch_size': 64,
    'hidden_size': 256,
    'head_size': 16,
    'num_heads': 16,
    'intermediate_size': 256 * 8,
    'hidden_dropout_prob': 0.01,
    'attention_probs_dropout_prob': 0.01,
    'num_data_workers': 4,
    'nums_node_feat_cate': [i+1 for i in features.ATOM_CATE_FEAT_DIMS] + [20],
    'num_node_feat_float': features.ATOM_FLOAT_FEAT_DIM,
    'num_spread_layers': 8,
    'nums_structure_feat_cate': 
        [i+1 for i in features.BOND_CATE_FEAT_DIMS] + [2] + [
            features.MAX_SHORTEST_PATH_LEN + 2, features.MAX_SAME_RING_COUNT+1, features.MAX_SAME_RING_MIN_SIZE+1,
        ] + [4],
    'num_structure_feat_float': features.ATOM_FLOAT_FEAT_DIM + features.BOND_FLOAT_FEAT_DIM ,
    'learning_rate_decay_rate': 0.97,
    'learning_rate': 1e-4,
    'learning_rate_min': 5e-7,
    'warmup_epochs': 0,
    'weight_decay': 1e-8,
    'data_cache_path': os.path.expanduser('~/pcqm4m_cache.pkl'),
    'middle_data_path': os.path.expanduser('~/data/zouxiaochuan/middle_data/pcqm4m'),
    'num_cluster': 3,
    'temperature': 0.1
}
