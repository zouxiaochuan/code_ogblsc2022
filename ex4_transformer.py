from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import transformers


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
            nn.Linear(hidden_size, self.attention_head_size))
        self.structure_map_value = nn.Sequential(
            nn.Linear(hidden_size, self.attention_head_size))

        self.attention_weight = nn.Parameter(
            nn.init.uniform_(
                torch.ones(4, dtype=torch.float32)))
        self.triplet_map_query = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.triplet_map_key = nn.Linear(self.attention_head_size, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_matrix=None,
        triple_matrix=None,
    ):

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        # query_layer: [B, H, L, D]
        
        if structure_matrix is not None:
            structure_value = self.structure_map_value(structure_matrix)
            structure_query = self.structure_map_query(structure_matrix)
            # structure_query: [B, L, L, D]

            query_layer = query_layer[:, :, :, None, :] + \
                query_layer[:, :, None, :, :] + structure_query[:, None, :, :, :]
            
            value_layer = value_layer[:, :, :, None, :] + \
                value_layer[:, :, None, :, :] + structure_value[:, None, :, :, :]
            pass
        else:
            query_layer = query_layer[:, :, :, None, :] + \
                query_layer[:, :, None, :, :]
            
            value_layer = value_layer[:, :, :, None, :] + \
                value_layer[:, :, None, :, :]
            pass            
        
        attention_scores = torch.einsum('bhijd,bhkd->bhijk', query_layer, key_layer)

        triplet_query = self.triplet_map_query(triple_matrix)
        triplet_key = self.triplet_map_key(triple_matrix)
        triplet_scores_query = torch.einsum('bijkd,bhijd->bhijk', triplet_query, query_layer)
        triplet_scores_key = torch.einsum('bijkd,bhkd->bhijk', triplet_key, key_layer)
        attention_scores += triplet_scores_query + triplet_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        B, H, L, _, _ = attention_scores.shape
        attention_scores = attention_scores.view(B, H, L*L, -1)

        if attention_mask is not None:
            # attention_mask: [B, 1, 1, L]
            attention_mask_ = attention_mask.reshape(B, L)
            attention_mask_ = attention_mask_[:, None, None, :] * \
                attention_mask_[:, None, :, None] * attention_mask_[:, :, None, None]
            attention_mask_ = attention_mask_.reshape(B, 1, L*L, -1)
            attention_scores = attention_scores + attention_mask_

        # attention_probs = nn.functional.relu(attention_scores) + 1e-12
        # attention_probs = attention_scores / (attention_scores.sum(dim=-1, keepdim=True) + 1e-12)
        attention_probs = nn.functional.softmax(attention_scores, dim=-2)
        attention_probs = self.dropout(attention_probs)

        value_layer = value_layer.reshape(B, H, L*L, -1)
        context_layer = torch.matmul(attention_probs.transpose(-1, -2), value_layer)

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
            hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm_output = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_matrix=None,
        triplet_matrix=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            structure_matrix=structure_matrix,
            triple_matrix=triplet_matrix,
        )

        output = self.dense(self_outputs)
        output = self.dropout(output)
        output = self.layer_norm_output(output + hidden_states)
        return output

    pass



class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob,
                 num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        # intermediate_size = hidden_size
        self.attention = Attention(
            hidden_size, hidden_dropout_prob, num_attention_heads,
            attention_probs_dropout_prob)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_output = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm_output = nn.LayerNorm(hidden_size)
        self.dropout_output = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_matrix=None,
        triplet_matrix=None,
    ):     
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            structure_matrix=structure_matrix,
            triplet_matrix=triplet_matrix,
        )
        
        attention_output = self_attention_outputs
        x_intermidiate = self.intermediate_act_fn(self.intermediate(attention_output))
        output = self.dense_output(x_intermidiate)
        output = self.dropout_output(output)
        output = self.layer_norm_output(output + attention_output)
        return output
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