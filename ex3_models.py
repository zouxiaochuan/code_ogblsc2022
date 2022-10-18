import torch
import torch.nn as nn
import simple_transformer as transformer


class CateFeatureEmbedding(nn.Module):
    def __init__(self, num_uniq_values, embed_dim):
        '''
        '''
        super().__init__()
        csum = torch.cumsum(torch.LongTensor(num_uniq_values), dim=0)
        num_emb = csum[-1]
        num_uniq_values = torch.LongTensor(num_uniq_values).reshape(1, 1, -1)
        self.register_buffer('num_uniq_values', num_uniq_values)
        
        starts = torch.cat(
            (torch.LongTensor([0]), csum[:-1])).reshape(1, -1)
        self.register_buffer('starts', starts)
        
        self.embeddings = nn.Embedding(
            num_emb, embed_dim)
        
        self.layer_norm_output = nn.LayerNorm(embed_dim)
        pass

    def forward(self, x):
        if torch.any(x < 0):
            raise RuntimeError(str(x))
        
        if torch.any(torch.ge(x, self.num_uniq_values)):
            raise RuntimeError(str(x))
            pass
        
        x = x + self.starts

        emb = self.embeddings(x).sum(dim=-2)

        return self.layer_norm_output(emb)
    pass


class FloatFeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        '''
        '''
        super().__init__()
        self.embeddings = nn.Linear(input_dim, embed_dim)
        self.layer_norm_output = nn.LayerNorm(embed_dim)
        pass

    def forward(self, x):
        emb = self.embeddings(x)

        return self.layer_norm_output(emb)
    pass



class StructureEmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config['head_size']
        self.hidden_size = config['hidden_size']

        self.virtual_edge_emb = nn.Parameter(
            nn.init.normal_(torch.empty(1, 1, self.hidden_size)))

        self.layer_structure_embed_cate = CateFeatureEmbedding(
            config['nums_structure_feat_cate'], self.hidden_size)
        
        self.layer_structure_embed_float = FloatFeatureEmbedding(
            config['num_structure_feat_float'], self.hidden_size)
        self.layer_bond_reverse = nn.Linear(self.hidden_size, self.hidden_size)
        pass

    def forward(self, structure_feat_cate, structure_feat_float):
        max_node = structure_feat_cate.shape[1] + 1
        batch_size = structure_feat_cate.shape[0]
        device = structure_feat_cate.device
        hidden_structure = torch.zeros((batch_size, max_node, max_node, self.hidden_size), device=device)
        hidden_structure[:, 0, :, :] = self.virtual_edge_emb.expand(batch_size, max_node, -1)
        hidden_structure[:, :, 0, :] = self.virtual_edge_emb.expand(batch_size, max_node, -1)
        hidden_structure[:, 1:, 1:, :] = self.layer_structure_embed_cate(structure_feat_cate) + \
            self.layer_structure_embed_float(structure_feat_float)

        return hidden_structure
        

class SpreadLayer(nn.Module):
    def __init__(self, config):
        '''
        '''
        super().__init__()
        
        self.layer_spread = transformer.TransformerLayer(
            config['hidden_size'], intermediate_size=config['intermediate_size'],
            attention_head_size=config['head_size'], num_attention_heads=config['num_heads'],
            hidden_dropout_prob=config['hidden_dropout_prob'],
            attention_probs_dropout_prob=config['attention_probs_dropout_prob'],
        )
        
        pass

    def forward(self, hidden_node, node_mask, structure_matrix, triplet_matrix):
        
        return self.layer_spread(hidden_node, node_mask, structure_matrix)
        # return self.layer_spread(hidden_node, node_mask)
        pass


class MoleculeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.head_size = config['head_size']
        self.num_spread_layers = config['num_spread_layers']
        self.num_attention_heads = self.hidden_size // self.head_size
        
        self.layer_node_embed_cate = CateFeatureEmbedding(
                config['nums_node_feat_cate'], self.hidden_size)
        
        self.layer_node_embed_float = FloatFeatureEmbedding(
                config['num_node_feat_float'], self.hidden_size)


        self.virtual_node_emb = nn.Parameter(
            nn.init.normal_(torch.empty(1, 1, self.hidden_size)))


        self.layer_output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

        self.spread_layers = nn.ModuleList(
            [SpreadLayer(config) for _ in range(self.num_spread_layers)])
        
        self.layer_structure_emb = StructureEmbeddingLayer(config)
        pass

    def forward(
            self, graph):
        
        node_feat_cate = graph['node_feat_cate']
        device = node_feat_cate.device
        node_feat_float = graph['node_feat_float']
        node_mask = graph['node_mask']
        structure_feat_cate = graph['structure_feat_cate']
        structure_feat_float = graph['structure_feat_float']

        hidden_node = self.layer_node_embed_cate(node_feat_cate)
        hidden_node = hidden_node + \
            self.layer_node_embed_float(node_feat_float)

        max_node = hidden_node.shape[1] + 1

        # node 0 is virtual node
        batch_size = hidden_node.shape[0]
        
        hidden_node = torch.cat(
            (self.virtual_node_emb.expand(batch_size, -1, -1), hidden_node), dim=1)
        # hidden_node: [B, max_node, hidden_size]

        node_mask = torch.cat(
            (torch.ones((batch_size, 1), device=device),
             node_mask), dim=-1)

        # hidden_bond = self.layer_bond_embed_cate(bond_feat_cate)
        # hidden_bond = hidden_bond + self.layer_bond_embed_float(bond_feat_float)
        # hidden_bond *= bond_mask[..., None]

        # bond_index += 1
        # batch_range = torch.arange(batch_size, device=device)[:, None]

        # hidden_node[batch_range.expand(-1, bond_index.shape[-1]), bond_index[:, 1, :]] += hidden_bond
        
        structure_matrix = self.layer_structure_emb(structure_feat_cate, structure_feat_float)
        node_mask = (1.0 - node_mask[:, None, None, :]) * -10000000

        # triplet_matrix = self.layer_triplet_emb(triplet_feat_cate)

        for layer in self.spread_layers:
            hidden_node = layer(
                hidden_node, node_mask, structure_matrix, None)
            pass
        
        return hidden_node
        pass

class MoleculeHLGapPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_layer = MoleculeEmbedding(config)
        self.layer_output = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.Tanh(),
            nn.Linear(config['hidden_size'], 1)
        )
        pass

    def forward(
            self, graph):

        hidden_node = self.embed_layer(
            graph)
        
        return self.layer_output(hidden_node[:, 0, :]).flatten()
        pass

class MoleculePairDistPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_layer = MoleculeEmbedding(config)
        # self.layer_output = nn.Sequential(
        #     nn.Linear(config['hidden_size'], config['hidden_size']),
        #     nn.Tanh(),
        #     nn.Linear(config['hidden_size'], config['hidden_size']),
        # )
        self.layer_output = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.Tanh(),
            nn.Linear(config['hidden_size'], config['hidden_size'])
        )
        pass

    def forward(
            self, graph):

        hidden_node = self.embed_layer(
            graph)
        
        hidden_node = hidden_node[:, 1:, :]
        hidden_node = self.layer_output(hidden_node)
        hidden_node = hidden_node[:, :graph['num_atom'], :]
        dist = torch.cdist(hidden_node, hidden_node, p=2)
        # dist = torch.einsum('bid,bjd->bij', hidden_node, hidden_node)
        return dist
        pass
