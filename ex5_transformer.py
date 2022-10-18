from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import transformers

class SelfAttentionEdge2Node(nn.Module):
    def __init__(self, hidden_size, attention_head_size, num_attention_heads=1, attention_probs_dropout_prob=0.1, hidden_dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(3*hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense_output = nn.Linear(hidden_size, hidden_size)
        self.drop_output = nn.Dropout(hidden_dropout)
        self.layer_norm_output = nn.LayerNorm(hidden_size)

        self.triplet_query = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.triplet_key = nn.Linear(self.attention_head_size, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (-1, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        if x.dim() == 5:
            return x.permute(0, 3, 1, 2, 4).contiguous()
        elif x.dim() == 4:
            return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states,
        structure_matrix,
        triplet_matrix,
        attention_mask=None,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        B, H, N, D = query_layer.shape
        # query_layer: [B, H, N, D]
        key_layer = torch.cat(
            [
                torch.tile(hidden_states[:, :, None, :], (1, 1, N, 1)),
                torch.tile(hidden_states[:, None, :, :], (1, N, 1, 1)),
                structure_matrix,
            ],
            dim=-1
        )
        key_layer = self.transpose_for_scores(self.key(key_layer))
        # key_layer: [B, H, N, N, D]

        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.einsum('bhnd,bhlrd->bhlrn', query_layer, key_layer)
        # attention_scores: [B, H, N, N, N]

        triplet_query_layer = self.triplet_query(triplet_matrix)
        triplet_key_layer = self.triplet_key(triplet_matrix)
        attention_scores_triplet_query = torch.einsum('bhnd,blrnd->bhlrn', query_layer, triplet_query_layer)
        attention_scores_triplet_key = torch.einsum('bhlrd,blrnd->bhlrn', key_layer, triplet_key_layer)
        attention_scores = attention_scores + attention_scores_triplet_query + attention_scores_triplet_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores /= 4.0

        attention_mask_ = attention_mask[:, None, :, :, :]
        attention_scores = attention_scores + attention_mask_

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.einsum('bhlrn,bhnd->bhlrd', attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape).contiguous()

        outputs = self.dense_output(context_layer)
        outputs = self.drop_output(outputs)
        outputs = self.layer_norm_output(outputs + structure_matrix)

        return outputs

    pass


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, attention_head_size, num_attention_heads,
            attention_probs_dropout_prob=0.1):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.structure_map_query = nn.Sequential(
            nn.Linear(hidden_size, self.attention_head_size))
        self.structure_map_key = nn.Sequential(
            nn.Linear(hidden_size, self.attention_head_size))


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (-1, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states,
        structure_matrix,
        attention_mask
    ):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if structure_matrix is not None:
            structure_key = self.structure_map_key(structure_matrix)
            structure_query = self.structure_map_query(structure_matrix)
            structure_scores_query = torch.einsum("bhld,blrd->bhlr", query_layer, structure_query)
            structure_scores_key = torch.einsum("bhrd,blrd->bhlr", key_layer, structure_key)
            attention_scores = attention_scores + structure_scores_query + \
                structure_scores_key
            pass
        
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores /= 4.0

        # if attention_mask is not None:
        #     attention_scores = attention_scores + attention_mask

        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # B, H, N, _ = attention_scores.shape
        # attention_scores = attention_scores.reshape(B, 4, -1, N, N)
        # attention_scores0 = torch.sigmoid(attention_scores[:, 0, :, :, :])
        # attention_scores1 = F.relu(attention_scores[:, 1, :, :, :])
        # attention_scores2 = 1.0 / (1.0 + F.relu(-attention_scores[:, 2, :, :, :]))
        # attention_scores3 = torch.pow(F.relu(attention_scores[:, 3, :, :, :]), 2)

        # attention_scores = torch.stack(
        #     [attention_scores0, attention_scores1, attention_scores2, attention_scores3], dim=1).reshape(B, H, N, N)

        if attention_mask is not None:
            # attention_scores = attention_scores * (attention_mask==0).float()      
            attention_scores = attention_scores + attention_mask
            # attention_scores = attention_scores + \
            #     torch.eye(attention_scores.shape[-1], device=attention_scores.device)[None, None, :, :] * -100000.0

        # attention_probs = nn.functional.relu(attention_scores) + 1e-12
        # attention_probs = attention_scores / (attention_scores.sum(dim=-1, keepdim=True) + 1e-12)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape).contiguous()

        outputs = context_layer

        return outputs

    pass



class Attention(nn.Module):
    def __init__(self, hidden_size, attention_head_size, num_attention_heads, 
            hidden_dropout_prob, attention_probs_dropout_prob):
        super().__init__()
        self.self_node2edge = SelfAttentionEdge2Node(
            hidden_size, attention_head_size, num_attention_heads, attention_probs_dropout_prob)
        self.self = SelfAttention(
            hidden_size, attention_head_size, num_attention_heads, attention_probs_dropout_prob)
        self.dense = nn.Linear(num_attention_heads * attention_head_size, hidden_size)
        # self.dense2 = nn.Linear(hidden_size*8, hidden_size)
        self.layer_norm_output = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)


    def forward(
        self,
        hidden_states,
        structure_matrix,
        triplet_matrix,
        attention_mask,
    ):
        structure_matrix = self.self_node2edge(hidden_states, structure_matrix, triplet_matrix, attention_mask)
        self_outputs = self.self(
            hidden_states,
            structure_matrix,
            attention_mask
        )

        output = self.dense(self_outputs)
        # output = F.gelu(output)
        # output = output + self_outputs.tile([1, 1, 4])
        # output = self.dense2(output)
        output = self.dropout(output)
        output = self.layer_norm_output(output + hidden_states)
        return output, structure_matrix

    pass



class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, attention_head_size, num_attention_heads,
            hidden_dropout_prob, attention_probs_dropout_prob):
        super().__init__()
        # intermediate_size = hidden_size
        self.attention = Attention(
            hidden_size, attention_head_size, num_attention_heads,
            hidden_dropout_prob, attention_probs_dropout_prob)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_output = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm_output = nn.LayerNorm(hidden_size)
        self.dropout_output = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        structure_matrix,
        triplet_matrix,
        attention_mask
    ):     
        node_outputs, structure_outputs = self.attention(
            hidden_states,
            structure_matrix,
            triplet_matrix,
            attention_mask
        )
        
        attention_output = node_outputs
        x_intermidiate = self.intermediate_act_fn(self.intermediate(attention_output))
        output = self.dense_output(x_intermidiate)
        output = self.dropout_output(output)
        output = self.layer_norm_output(output + attention_output)
        # output = attention_output
        return output, structure_outputs
    pass
