import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, static_features=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        if static_features is not None:
            src = src + static_features
            
        return src

class DynamicEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, static_features):
        for layer in self.layers:
            x = layer(x, static_features)
        return x

class TransformerBasedGatingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, output_dim)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        original_shape = x.shape  
        input_dim = x.shape[-1]
        
        if len(original_shape) == 2:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x.reshape(-1, 1, input_dim)
        
        batch_size_seq, seq_len, _ = x_seq.shape
        
        x_proj = self.input_proj(x_seq)
        
        pos_encoding_expanded = self.pos_encoding[:, :seq_len, :]
        x_proj = x_proj + pos_encoding_expanded
        
        x_transformed = x_proj
        for layer in self.transformer_layers:
            x_transformed = layer(x_transformed)
        
        x_transformed = self.layer_norm(x_transformed)
        
        gate_logits = self.output_proj(x_transformed)
        
        if len(original_shape) == 2:
            gate_logits = gate_logits.squeeze(1)
        else:
            gate_logits = gate_logits.reshape(*original_shape[:-1], self.output_dim)
        
        gate_weights = torch.sigmoid(gate_logits)
        
        return gate_weights, gate_logits

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.dropout(last_hidden)

class TemporalLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(TemporalLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return self.dropout(output)

class DimensionWiseGatingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=768, nhead=8, num_layers=3):
        super().__init__()
        self.gating_network = TransformerBasedGatingNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

    def forward(self, x):
        gate_weights, gate_logits = self.gating_network(x)
        return gate_weights, gate_logits

class MultimodalSoftGate(nn.Module):
    def __init__(self, modal_dims, hidden_size=256, target_sparsity=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_sparsity = target_sparsity

        self.task_embeddings = nn.ParameterDict()
        for modal_name in modal_dims.keys():
            embedding = torch.empty(hidden_size)
            nn.init.uniform_(embedding, -1.0, 1.0)
            self.task_embeddings[modal_name] = nn.Parameter(embedding)

        self.modal_gates = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.modal_gates[modal_name] = DimensionWiseGatingNetwork(
                input_dim=dim + hidden_size,
                output_dim=dim
            )
    
    def forward(self, modal_features):
        masked_features = {}
        gate_weights_dict = {}
        total_sparsity = 0

        for modal_name, features in modal_features.items():
            original_shape = features.shape
            is_temporal = len(original_shape) == 3

            if is_temporal:
                batch_size, timesteps, hidden_dim = original_shape

                features_flat = features.reshape(-1, hidden_dim)

                task_emb = self.task_embeddings[modal_name]
                task_emb_expanded = task_emb.unsqueeze(0).expand(batch_size * timesteps, -1)
                combined = torch.cat([features_flat, task_emb_expanded], dim=-1)

                gate_weights_flat, gate_logits = self.modal_gates[modal_name](combined)

                sparsity_modal = compute_sparsity_regularization(gate_weights_flat, self.target_sparsity)
                total_sparsity += sparsity_modal

                masked_feat_flat = gate_weights_flat * features_flat

                masked_features[modal_name] = masked_feat_flat.reshape(original_shape)
                gate_weights_dict[modal_name] = gate_weights_flat.reshape(original_shape)
            else:
                batch_size = features.size(0)

                task_emb = self.task_embeddings[modal_name]
                task_emb_expanded = task_emb.unsqueeze(0).expand(batch_size, -1)

                combined = torch.cat([features, task_emb_expanded], dim=-1)
                
                gate_weights, gate_logits = self.modal_gates[modal_name](combined)

                sparsity_modal = compute_sparsity_regularization(gate_weights, self.target_sparsity)
                total_sparsity += sparsity_modal
                
                masked_feat = gate_weights * features
                masked_features[modal_name] = masked_feat
                gate_weights_dict[modal_name] = gate_weights

        return masked_features, gate_weights_dict, total_sparsity

class ModalInterSoftGating(nn.Module):
    def __init__(self, modal_dims, hidden_dim, task_aware=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_aware = task_aware

        self.soft_gate = MultimodalSoftGate(modal_dims, hidden_dim)

    def forward(self, modal_features, task_type=None):
        gated_features, gate_weights, kl_loss = self.soft_gate(modal_features)

        gate_info = {
            'modal_gate_weights': gate_weights,
            'kl_loss': kl_loss
        }

        return gated_features, gate_info

class StaticQueryCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, static_features: torch.Tensor, dynamic_features: torch.Tensor):
        batch_size, num_timesteps, _ = dynamic_features.shape

        attended_dynamic, attention_weights = self.cross_attention(
            query=static_features,
            key=dynamic_features,
            value=dynamic_features
        )

        enhanced_dynamic = self.layer_norm(static_features + attended_dynamic)
        return enhanced_dynamic, attention_weights.squeeze(1)

class MELT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.modal_num = 5
        self.drop_ratio = config.drop_ratio
        self.task = config.task

        self.modal_encoders = nn.ModuleDict({
            "triage": nn.Sequential(
                nn.Linear(5, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.drop_ratio)
            ),
            "labtest": TemporalLSTMEncoder(59, self.hidden_dim, self.drop_ratio),
            "medications": TemporalLSTMEncoder(406, self.hidden_dim, self.drop_ratio),
            "diagnoses": nn.Sequential(
                nn.Linear(5741, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.drop_ratio)
            ),
            "notes": nn.Sequential(
                nn.Linear(768, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.drop_ratio)
            )
        })

        modal_dims = {
            'triage': self.hidden_dim,
            'labtest': self.hidden_dim, 
            'medications': self.hidden_dim,
            'diagnoses': self.hidden_dim,
            'notes': self.hidden_dim
        }
        self.modal_inter_gating = ModalInterSoftGating(modal_dims, self.hidden_dim, task_aware=True)

        self.static_dynamic_interaction = StaticQueryCrossAttention(self.hidden_dim)

        self.dynamic_encoder = DynamicEncoder(
            d_model=self.hidden_dim,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=self.drop_ratio
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        self.fusion_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.residual_weight = nn.Parameter(torch.tensor(0.0))

        if self.task == 'ihm':
            self.out_proj = nn.Linear(self.hidden_dim, 2)
        elif self.task == 'los':
            self.out_proj = nn.Linear(self.hidden_dim, 1)
        else:
            raise ValueError(f"Unknown task: {config.task}")

    def forward(self, inputs):
        bz = inputs["triage"].size(0)
        
        modal_features = {}

        triage_feat = self.modal_encoders['triage'](inputs['triage'])
        modal_features['triage'] = triage_feat

        labtest_output = self.modal_encoders['labtest'](inputs['labtest'])
        modal_features["labtest"] = labtest_output

        medications_output = self.modal_encoders['medications'](inputs['medications'])
        modal_features["medications"] = medications_output

        diagnoses_feat = self.modal_encoders['diagnoses'](inputs['diagnoses'].squeeze(1))
        modal_features['diagnoses'] = diagnoses_feat

        notes_feat = self.modal_encoders['notes'](inputs['notes'].squeeze(1))
        modal_features['notes'] = notes_feat

        gated_features, gate_info = self.modal_inter_gating(modal_features)

        gated_labtest = torch.mean(gated_features['labtest'], dim=1)
        gated_medications = torch.mean(gated_features['medications'], dim=1) 

        all_features = torch.stack([
            gated_features['triage'], 
            gated_labtest, 
            gated_medications, 
            gated_features['diagnoses'], 
            gated_features['notes']
        ], dim=1)
        attention_weights = self.attention_pool(all_features)
        final_features = torch.sum(all_features * attention_weights, dim=1)

        global_static_features = (gated_features['triage'] + gated_features['diagnoses'] + gated_features['notes']) / 3.0
        static_features = torch.cat([gated_features['triage'].unsqueeze(1), 
                                       gated_features['diagnoses'].unsqueeze(1), 
                                       gated_features['notes'].unsqueeze(1)], dim=1)
        
        dynamic_lab = gated_features['labtest']
        dynamic_med = gated_features['medications']

        dynamic_features = torch.cat([dynamic_lab, dynamic_med], dim=1)

        enhanced_s2d, cross_attention_weights = self.static_dynamic_interaction(
            static_features, dynamic_features
        )
        enhanced_s2d_agg = torch.mean(enhanced_s2d, dim=1)

        global_static_features_expanded = global_static_features.unsqueeze(1).repeat(1, dynamic_features.size(1), 1)

        enhanced_d2s = self.dynamic_encoder(dynamic_features, global_static_features_expanded)

        enhanced_d2s_agg = torch.mean(enhanced_d2s, dim=1)
        final_combined = self.fusion_layer(torch.cat([enhanced_s2d_agg, enhanced_d2s_agg], dim=-1))

        final_combined = self.residual_weight * final_combined + final_features

        outputs = self.out_proj(final_combined)

        gate_info.update({
            'cross_attention_weights': cross_attention_weights
        })

        return outputs, gate_info

class TaskAwareGating(nn.Module):
    def __init__(self, hidden_dim, num_timesteps=24):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.task_prompt = nn.Parameter(torch.randn(hidden_dim))
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, temporal_features):
        batch_size = temporal_features.size(0)
        assert(temporal_features.size(1) == self.num_timesteps)

        task_prompt_expanded = self.task_prompt.unsqueeze(0).unsqueeze(1)
        task_prompt_expanded = task_prompt_expanded.repeat(batch_size, self.num_timesteps, 1)

        concatenated = torch.cat([temporal_features, task_prompt_expanded], dim=-1)

        gate_weights = self.gate_mlp(concatenated)

        weighted_features = temporal_features * gate_weights

        attention_scores = self.attention_pool(weighted_features)
        final_features = torch.sum(weighted_features * attention_scores, dim=1)

        return final_features, attention_scores.squeeze(-1)

def compute_bernoulli_kl(gate_weights, prior_prob=0.5, epsilon=1e-8):
    pi = torch.clamp(gate_weights, epsilon, 1-epsilon)
    prior_prob_tensor = torch.full_like(pi, prior_prob)
    prior_prob_tensor = torch.clamp(prior_prob_tensor, epsilon, 1-epsilon)
    
    term1 = pi * (torch.log(pi + epsilon) - torch.log(prior_prob_tensor + epsilon))
    term2 = (1 - pi) * (torch.log(1 - pi + epsilon) - torch.log(1 - prior_prob_tensor + epsilon))
    
    kl = term1 + term2
    kl = torch.where(torch.isnan(kl), torch.zeros_like(kl), kl)
    
    return kl.mean()

def compute_sparsity_regularization(gate_weights, target_sparsity=0.8, epsilon=1e-8):
    current_sparsity = torch.mean((gate_weights < 0.1).float())
    
    l1_loss = torch.mean(torch.abs(gate_weights))
    
    entropy = -gate_weights * torch.log(gate_weights + epsilon) - (1 - gate_weights) * torch.log(1 - gate_weights + epsilon)
    entropy_loss = torch.mean(entropy)
    
    sparsity_match_loss = torch.abs(current_sparsity - target_sparsity)
    
    total_loss = (
        0.7 * l1_loss +
        0.3 * entropy_loss +
        0.3 * sparsity_match_loss
    )
    
    scaling_factor = 0.1
    return total_loss * scaling_factor

def compute_multi_scale_sparsity(gate_weights_dict, target_sparsity=0.5):
    total_loss = 0
    modal_count = 0
    
    for modal_name, gate_weights in gate_weights_dict.items():
        if modal_name in ['labtest', 'medications']:
            modal_target_sparsity = target_sparsity * 0.8
        elif modal_name in ['diagnoses', 'notes']:
            modal_target_sparsity = target_sparsity * 1.2
        else:
            modal_target_sparsity = target_sparsity
        
        modal_loss = compute_sparsity_regularization(gate_weights, modal_target_sparsity)
        total_loss += modal_loss
        modal_count += 1
    
    return total_loss / max(modal_count, 1)

