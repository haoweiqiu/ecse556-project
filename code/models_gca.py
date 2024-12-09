from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)

import os

class BIN_Interaction_Flat(nn.Module):
    """
    Interaction Network with 2D interaction map using Gated Cross-Attention (GCA)
    """

    def __init__(self, **config):
        super(BIN_Interaction_Flat, self).__init__()
        self.max_d = config["max_drug_seq"]
        self.max_p = config["max_protein_seq"]
        self.emb_size = config["emb_size"]
        self.dropout_rate = config["dropout_rate"]

        # Manually set key_dim
        self.key_dim = 64  # Common manually assigned value for key_dim

        # DenseNet configuration
        self.flatten_dim = 768
        self.batch_size = config["batch_size"]
        self.input_dim_drug = config["input_dim_drug"]
        self.input_dim_target = config["input_dim_target"]

        # Embedding layers
        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)

        # Encoders
        self.d_encoder = Encoder_MultipleLayers(
            2, self.emb_size, config["intermediate_size"], config["num_attention_heads"],
            config["attention_probs_dropout_prob"], config["hidden_dropout_prob"]
        )
        self.p_encoder = Encoder_MultipleLayers(
            2, self.emb_size, config["intermediate_size"], config["num_attention_heads"],
            config["attention_probs_dropout_prob"], config["hidden_dropout_prob"]
        )

        # Gated Cross-Attention module
        self.gca = GatedCrossAttention(
            hidden_size=self.emb_size,
            key_dim=self.key_dim,  # Use the manually assigned key_dim
            max_drug_seq=self.max_d,
            max_protein_seq=self.max_p,
            dropout_rate=self.dropout_rate,
            save_attention_maps=True,  # Enable saving attention maps
            map_save_dir="./attention_maps"  # Save location
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)  # Output layer
        )

    def forward(self, d, p, d_mask, p_mask, batch_index=0):
        # Mask expansion
        ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2)
        ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0
        print(f"ex_d_mask shape: {ex_d_mask.shape}")
        print(f"ex_p_mask shape: {ex_p_mask.shape}")

        # Embedding layers
        d_emb = self.demb(d)
        p_emb = self.pemb(p)
        print(f"d_emb shape: {d_emb.shape}")
        print(f"p_emb shape: {p_emb.shape}")

        # Encoders
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
        p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())
        print(f"d_encoded_layers shape: {d_encoded_layers.shape}")
        print(f"p_encoded_layers shape: {p_encoded_layers.shape}")

        # Gated Cross-Attention
        V_d_prime, V_p_prime = self.gca(d_encoded_layers, p_encoded_layers, batch_index=batch_index)
        print(f"V_d_prime shape: {V_d_prime.shape}")
        print(f"V_p_prime shape: {V_p_prime.shape}")

        # Pooling operation
        d_pooled = torch.mean(V_d_prime, dim=1)  # Average pooling over the sequence length
        p_pooled = torch.mean(V_p_prime, dim=1)
        print(f"d_pooled shape: {d_pooled.shape}")
        print(f"p_pooled shape: {p_pooled.shape}")

        # Flatten and pass to decoder
        flattened = torch.cat((d_pooled, p_pooled), dim=1)
        print(f"flattened shape: {flattened.shape}")
        score = self.decoder(flattened)
        print(f"score shape: {score.shape}")

        return score


class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention Module with Residual Connections, Layer Normalization, and Debug Prints
    """

    def __init__(self, hidden_size, key_dim, max_drug_seq, max_protein_seq, dropout_rate, save_attention_maps=False, map_save_dir="./attention_maps"):
        super(GatedCrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.max_drug_seq = max_drug_seq
        self.max_protein_seq = max_protein_seq
        self.dropout_rate = dropout_rate
        self.save_attention_maps = save_attention_maps  # Flag to enable saving
        self.map_save_dir = map_save_dir  # Directory to save maps

        if self.save_attention_maps and not os.path.exists(self.map_save_dir):
            os.makedirs(self.map_save_dir)

        # Query, Key, Value projections
        self.query_projection_drug = nn.Linear(hidden_size, key_dim)
        self.key_projection_protein = nn.Linear(hidden_size, key_dim)
        self.value_projection_drug = nn.Linear(hidden_size, hidden_size)

        self.query_projection_protein = nn.Linear(hidden_size, key_dim)
        self.key_projection_drug = nn.Linear(hidden_size, key_dim)
        self.value_projection_protein = nn.Linear(hidden_size, hidden_size)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, drug_emb, protein_emb, batch_index=0):
        print(f"Input drug_emb shape: {drug_emb.shape}")
        print(f"Input protein_emb shape: {protein_emb.shape}")

        # Protein-to-Drug Attention
        Q_p = self.query_projection_protein(protein_emb)
        print(f"Q_p shape: {Q_p.shape}")
        K_d = self.key_projection_drug(drug_emb)
        print(f"K_d shape: {K_d.shape}")
        V_d = self.value_projection_drug(drug_emb)
        print(f"V_d shape: {V_d.shape}")
        attention_scores_p_to_d = torch.matmul(Q_p, K_d.transpose(-1, -2)) / math.sqrt(self.key_dim)
        print(f"attention_scores_p_to_d shape: {attention_scores_p_to_d.shape}")
        a_p_to_d = F.softmax(attention_scores_p_to_d.sum(dim=1) / self.max_protein_seq, dim=-1)
        print(f"a_p_to_d shape: {a_p_to_d.shape}")
        V_d_prime = a_p_to_d.unsqueeze(-1) * V_d  # V_d'
        V_d_prime = self.layer_norm(V_d_prime + V_d)  # Residual connection and layer norm
        print(f"V_d_prime shape: {V_d_prime.shape}")

        # Save Protein-to-Drug Attention Map
        if self.save_attention_maps:
            map_path = os.path.join(self.map_save_dir, f"attention_map_p_to_d_batch_{batch_index}.pt")
            torch.save(attention_scores_p_to_d, map_path)
            print(f"Saved Protein-to-Drug attention map to: {map_path}")

        # Drug-to-Protein Attention
        Q_d = self.query_projection_drug(drug_emb)
        print(f"Q_d shape: {Q_d.shape}")
        K_p = self.key_projection_protein(protein_emb)
        print(f"K_p shape: {K_p.shape}")
        V_p = self.value_projection_protein(protein_emb)
        print(f"V_p shape: {V_p.shape}")
        attention_scores_d_to_p = torch.matmul(Q_d, K_p.transpose(-1, -2)) / math.sqrt(self.key_dim)
        print(f"attention_scores_d_to_p shape: {attention_scores_d_to_p.shape}")
        a_d_to_p = F.softmax(attention_scores_d_to_p.sum(dim=1) / self.max_drug_seq, dim=-1)
        print(f"a_d_to_p shape: {a_d_to_p.shape}")
        V_p_prime = a_d_to_p.unsqueeze(-1) * V_p  # V_p'
        V_p_prime = self.layer_norm(V_p_prime + V_p)  # Residual connection and layer norm
        print(f"V_p_prime shape: {V_p_prime.shape}")

        # Save Drug-to-Protein Attention Map
        if self.save_attention_maps:
            map_path = os.path.join(self.map_save_dir, f"attention_map_d_to_p_batch_{batch_index}.pt")
            torch.save(attention_scores_d_to_p, map_path)
            print(f"Saved Drug-to-Protein attention map to: {map_path}")

        return V_d_prime, V_p_prime  # Output final contextualized representations

   
# help classes    
    
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output    
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output    

    
class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states