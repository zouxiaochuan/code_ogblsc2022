import torch
import torch.nn as nn
import math
import transformers


class SelfAttentionPairDim1(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=1, attention_probs_dropout_prob=0.1, hidden_dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.key = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.value = nn.Linear(self.attention_head_size, self.attention_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense_output = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.drop_output = nn.Dropout(hidden_dropout)
        self.layer_norm_output = nn.LayerNorm(self.attention_head_size)

    def forward(
        self,
        hidden_states,
        structure_matrix,
        attention_mask=None
    ):
        query_layer = self.query(structure_matrix)[:, None, :, :, :].transpose(1, 2).contiguous()
        # query_layer: [B, 1, N, N, D]
        key_layer = self.key(structure_matrix)[:, :, :, None, :]
        # key_layer: [B, N, N, 1, D]

        
        value_layer = self.value(structure_matrix)[:, None, :, :, :].transpose(1, 2).contiguous()
        # value_layer: [B, N, 1, N, D]

        attention_scores = torch.sum(query_layer * key_layer, dim=-1)
        # attention_scores: [B, N, N, N]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores /= 4.0

        if attention_mask is not None:
            # attention_mask: [B, 1, 1, N]
            attention_scores = attention_scores + attention_mask

        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = nn.functional.relu(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.sum(attention_probs[:, :, :, :, None] * value_layer, dim=-2)

        outputs = self.dense_output(context_layer)
        outputs = self.drop_output(outputs)
        outputs = self.layer_norm_output(outputs + structure_matrix)

        return outputs

class SelfAttentionPairDim2(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=1, attention_probs_dropout_prob=0.1, hidden_dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.key = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.value = nn.Linear(self.attention_head_size, self.attention_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense_output = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.drop_output = nn.Dropout(hidden_dropout)
        self.layer_norm_output = nn.LayerNorm(self.attention_head_size)

    def forward(
        self,
        hidden_states,
        structure_matrix,
        attention_mask=None
    ):
        query_layer = self.query(structure_matrix)[:, :, None, :, :]
        # query_layer: [B, N, 1, N, D]
        key_layer = self.key(structure_matrix)[:, :, :, None, :]
        # key_layer: [B, N, N, 1, D]

        
        value_layer = self.value(structure_matrix)[:, :, None, :, :]
        # value_layer: [B, N, 1, N, D]

        attention_scores = torch.sum(query_layer * key_layer, dim=-1)
        # attention_scores: [B, N, N, N]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores /= 4.0

        if attention_mask is not None:
            # attention_mask: [B, 1, 1, N]
            attention_scores = attention_scores + attention_mask

        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = nn.functional.relu(attention_scores)
        attention_probs = attention_probs / torch.sum(attention_probs + 1e-12, dim=-1, keepdim=True)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.sum(attention_probs[:, :, :, :, None] * value_layer, dim=-2)

        outputs = self.dense_output(context_layer)
        outputs = self.drop_output(outputs)
        outputs = self.layer_norm_output(outputs + structure_matrix)

        return outputs


class SelfAttentionEdge2Node(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=1, attention_probs_dropout_prob=0.1, hidden_dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.key = nn.Linear(hidden_size, self.attention_head_size)
        self.value = nn.Linear(self.attention_head_size, self.attention_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense_output = nn.Linear(self.attention_head_size, hidden_size)
        self.drop_output = nn.Dropout(hidden_dropout)
        self.layer_norm_output = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states,
        structure_matrix,
        attention_mask=None
    ):
        query_layer = self.query(structure_matrix)
        # query_layer: [B, N, N, D]
        key_layer = self.key(hidden_states)
        # key_layer: [B, N, D]

        B, N, D = key_layer.shape
        value_layer = self.value(structure_matrix)
        attention_scores = torch.einsum('blrd,bnd->blrn', query_layer, key_layer)
        # attention_scores: [B, N, N, N]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores /= 4.0
        attention_scores = attention_scores.reshape(B, -1, N)

        if attention_mask is not None:
            # attention_mask: [B, 1, 1, N]
            attention_mask_ = attention_mask.reshape(B, N)
            attention_mask_2d = attention_mask_[:, :, None] * attention_mask_[:, None, :]
            attention_mask_2d = attention_mask_2d.reshape(B, -1)[:, :, None]
            attention_scores = attention_scores + attention_mask_2d
            pass

        # attention_probs = nn.functional.softmax(attention_scores, dim=-2)
        attention_probs = nn.functional.relu(attention_scores)
        attention_probs = attention_probs / torch.sum(attention_probs + 1e-12, dim=-1, keepdim=True)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.einsum('bmn,bmd->bnd', attention_probs, value_layer.reshape(B, -1, D))

        outputs = self.dense_output(context_layer)
        outputs = self.drop_output(outputs)
        outputs = self.layer_norm_output(outputs + hidden_states)

        return outputs

    pass

class SelfAttentionPair(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=1, attention_probs_dropout_prob=0.1, hidden_dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.attention_head_size)
        self.key = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.value = nn.Linear(hidden_size, self.attention_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense_output = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.drop_output = nn.Dropout(hidden_dropout)
        self.layer_norm_output = nn.LayerNorm(self.attention_head_size)

    def forward(
        self,
        hidden_states,
        structure_matrix,
        attention_mask=None
    ):
        query_layer = self.query(hidden_states)
        # query_layer: [B, N, D]
        key_layer = self.key(structure_matrix)
        # key_layer: [B, N, N, D]

        value_layer = self.value(hidden_states)
        attention_scores = torch.einsum('bnd,blrd->blrn', query_layer, key_layer)
        # attention_scores: [B, N, N, N]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores /= 4.0

        if attention_mask is not None:
            # attention_mask: [B, 1, 1, N]
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.einsum('blrn,bnd->blrd', attention_probs, value_layer)

        outputs = self.dense_output(context_layer)
        outputs = self.drop_output(outputs)
        outputs = self.layer_norm_output(outputs + structure_matrix)

        return outputs

    pass


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=1, attention_probs_dropout_prob=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.structure_map_query = nn.Sequential(
            nn.Linear(self.attention_head_size, self.attention_head_size))
        self.structure_map_key = nn.Sequential(
            nn.Linear(self.attention_head_size, self.attention_head_size))
        self.structure_map_value = nn.Sequential(
            nn.Linear(self.attention_head_size, self.attention_head_size))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_matrix=None,
    ):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # key_layer: [B, H, N, D]
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        if structure_matrix is not None:
            structure_key = self.structure_map_key(structure_matrix)
            structure_query = self.structure_map_query(structure_matrix)
            structure_value = self.structure_map_value(structure_matrix)

            key_layer = key_layer[:, :, :, None, :] + structure_key[:, None, :, :, : ]
            # key_layer: [B, H, N, N, D]

            value_layer = value_layer[:, :, None, :, :] + structure_value[:, None, :, :, : ]

            query_layer = query_layer[:, :, None, :, :] + structure_query[:, None, :, :, : ]
            pass
        else:
            key_layer = key_layer[:, :, :, None, :]
            value_layer = value_layer[:, :, None, :, :]
            query_layer = query_layer[:, :, None, :, :]
            pass

        attention_scores = torch.sum(key_layer * query_layer, dim=-1)
        # attention_scores: [B, H, N, N]
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = nn.functional.relu(attention_scores)
        attention_probs = attention_probs / torch.sum(attention_probs + 1e-12, dim=-1, keepdim=True)
        # attention_probs: [B, H, N, N]

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.sum(attention_probs[:, :, :, :, None] * value_layer, dim=-2)
        # context_layer: [B, H, N, D]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape).contiguous()

        outputs = context_layer

        return outputs

    pass



class Attention(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, num_attention_heads,
                 attention_probs_dropout_prob):
        super().__init__()
        self.self = SelfAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob=0.01)
        self.self_pair1 = SelfAttentionPairDim1(
            hidden_size, num_attention_heads, attention_probs_dropout_prob=0., hidden_dropout=0.)
        self.self_pair2 = SelfAttentionPairDim2(
            hidden_size, num_attention_heads, attention_probs_dropout_prob=0., hidden_dropout=0.)
        # self.self_e2n = SelfAttentionEdge2Node(
        #     hidden_size, num_attention_heads
        # )

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm_output = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_matrix=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            structure_matrix=structure_matrix,
        )
        
        structure_matrix = self.self_pair1(hidden_states, structure_matrix, attention_mask) + \
            self.self_pair2(hidden_states, structure_matrix, attention_mask)

        # self_outputs += self.self_e2n(hidden_states, structure_matrix, attention_mask)

        output = self.dense(self_outputs)
        output = self.dropout(output)
        output = self.layer_norm_output(output + hidden_states)
        return output, structure_matrix

    pass


class IntermediateLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        super().__init__()
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_output = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm_output = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        x_intermidiate = self.intermediate_act_fn(self.intermediate(hidden_states))
        output = self.dense_output(x_intermidiate)
        output = self.dropout(output)
        output = self.layer_norm_output(output + hidden_states)

        return output
        

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob,
                 num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        # intermediate_size = hidden_size
        head_size = hidden_size // num_attention_heads
        self.attention = Attention(
            hidden_size, hidden_dropout_prob, num_attention_heads,
            attention_probs_dropout_prob)
        self.intermediate_node = IntermediateLayer(hidden_size, intermediate_size, hidden_dropout_prob)
        self.intermediate_structure = IntermediateLayer(head_size, head_size*8, hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_matrix=None,
    ):     
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            structure_matrix=structure_matrix,
        )
        
        node_output, structure_output = self_attention_outputs
        node_output = self.intermediate_node(node_output)
        structure_output = self.intermediate_structure(structure_output)
        return node_output, structure_output
    pass


class Transformer(nn.Module):
    def __init__(self, num_layers, hidden_size, intermediate_size=None,
                 hidden_dropout_prob=0.1, num_attention_heads=4, attention_probs_dropout_prob=0.1,
                 reduction='top'):
        super().__init__()
        self.bias = nn.init.uniform_(
            nn.Parameter(torch.zeros(hidden_size)))
        
        self.num_attention_heads = num_attention_heads

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
            pass
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size, intermediate_size, hidden_dropout_prob, num_attention_heads,
                attention_probs_dropout_prob)
            for _ in range(num_layers)])

        self.reduction = reduction
        
        pass

    def forward(self, hidden_states, attention_mask=None, structure_matrix=None):

        hidden_states = self.forward_(hidden_states, attention_mask, structure_matrix)
        if self.reduction == 'top':
            return hidden_states[:, 0, :]
        elif self.reduction == 'mean':
            return torch.mean(hidden_states, dim=1)
        elif self.reduction is None:
            return torch.mean(hidden_states, dim=1), hidden_states
        else:
            raise RuntimeError(f'cannot recognize reduction: {self.reduction}')
        pass


    def forward_(self, hidden_states, attention_mask=None, structure_matrix=None):
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000000
            pass

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, structure_matrix)
            pass


        return hidden_states
    pass