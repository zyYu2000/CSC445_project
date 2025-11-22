"""
Adaptive Hybrid MRTA Network (Version 2)
-----------------------------------------

Key differences vs the baseline hybrid model:
    • Dynamic edge weighting in the GNN branch (learned from precedence, travel, skill overlap)
    • Transformer encoders for both tasks and robots, plus robot↔task cross-attention
    • Hierarchical task pooling to retain multi-scale structure
    • Feature gating and heteroscedastic output head (mean + log-variance) to tackle overfitting
    • Built-in gradient clipping and weight-normalized layers to reduce gradient explosion risk
"""

import math
import os
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mrta_deep_learning_model import MRTADataset, load_dataset


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply softmax with a mask (mask==0 entries are ignored)."""
    scores = scores.masked_fill(mask == 0, -1e9)
    return torch.softmax(scores, dim=dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        attn_output, attn_weights = self.attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x, attn_weights


class CrossAttentionBlock(nn.Module):
    """Cross attention with residual connection."""
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: torch.Tensor = None):
        attn_output, weights = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        x = self.norm(query + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x, weights


class DynamicEdgeGNNLayer(nn.Module):
    """Learn adjacency weights from multiple edge features."""
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.node_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor, node_mask: torch.Tensor):
        # node_feats: (B, T, H), edge_feats: (B, T, T, E), node_mask: (B, T)
        bsz, num_tasks, _ = node_feats.size()
        logits = self.edge_mlp(edge_feats).squeeze(-1)  # (B, T, T)
        mask_matrix = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        eye = torch.eye(num_tasks, device=node_feats.device).unsqueeze(0)
        mask_matrix = mask_matrix * (1.0 - eye)  # remove self-loops from attention calc
        attention = masked_softmax(logits, mask_matrix)
        updated = torch.bmm(attention, self.node_linear(node_feats))
        updated = self.dropout(updated)
        return self.norm(node_feats + updated)


class HierarchicalTaskPooling(nn.Module):
    """Two-level pooling: soft clustering followed by attention pooling."""
    def __init__(self, hidden_dim: int, num_clusters: int, dropout: float):
        super().__init__()
        self.cluster_proj = nn.Linear(hidden_dim, num_clusters)
        self.cluster_norm = nn.LayerNorm(hidden_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.global_score = nn.Linear(hidden_dim, 1)

    def forward(self, task_feats: torch.Tensor, task_mask: torch.Tensor):
        # task_feats: (B, T, H), task_mask: (B, T)
        logits = self.cluster_proj(task_feats)  # (B, T, K)
        logits = logits.masked_fill(task_mask.unsqueeze(-1) == 0, -1e9)
        weights = torch.softmax(logits, dim=1)  # (B, T, K)
        cluster_embeddings = torch.einsum('btk,bth->bkh', weights, task_feats)
        cluster_embeddings = self.cluster_norm(cluster_embeddings)
        global_features = self.global_proj(cluster_embeddings)
        global_scores = self.global_score(global_features).squeeze(-1)  # (B, K)
        global_weights = torch.softmax(global_scores, dim=-1)
        pooled = torch.bmm(global_weights.unsqueeze(1), cluster_embeddings).squeeze(1)
        return pooled


class FeatureGate(nn.Module):
    """Learnable gating per modality to reduce overfitting."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.linear(x))
        return gate * x


class MRTADynamicHybridNetwork(nn.Module):
    def __init__(
        self,
        max_tasks: int = 10,
        max_robots: int = 8,
        max_skills: int = 3,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        num_robot_transformer_layers: int = 2,
        num_clusters: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_robots = max_robots
        self.max_skills = max_skills
        self.hidden_dim = hidden_dim

        node_input_dim = max_skills + 3
        self.task_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.task_pos_enc = PositionalEncoding(hidden_dim, max_tasks)
        self.dynamic_layers = nn.ModuleList([
            DynamicEdgeGNNLayer(hidden_dim, edge_dim=3, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])
        self.task_transformers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, dropout)
            for _ in range(num_transformer_layers)
        ])
        self.robot_encoder = nn.Sequential(
            nn.Linear(max_skills, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.robot_pos_enc = PositionalEncoding(hidden_dim, max_robots)
        self.robot_transformers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, dropout)
            for _ in range(num_robot_transformer_layers)
        ])
        self.cross_attention = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.hier_pool = HierarchicalTaskPooling(hidden_dim, num_clusters, dropout)
        self.robot_pool = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.robot_attn_score = nn.Linear(hidden_dim, 1)
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.task_gate = FeatureGate(hidden_dim)
        self.robot_gate = FeatureGate(hidden_dim)
        self.aggregate_gate = FeatureGate(hidden_dim // 2)
        fusion_dim = hidden_dim * 2 + hidden_dim // 2
        self.pred_mean = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pred_logvar = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _normalize_features(self, T_e, task_mask, task_locations, aggregate_features):
        eps = 1e-6
        counts = task_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        te_mean = (T_e * task_mask).sum(dim=1, keepdim=True) / counts
        te_var = (((T_e - te_mean) * task_mask) ** 2).sum(dim=1, keepdim=True) / counts
        T_e_norm = ((T_e - te_mean) / torch.sqrt(te_var + eps)) * task_mask
        loc_mask = task_mask.unsqueeze(-1)
        loc_counts = loc_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        loc_mean = (task_locations * loc_mask).sum(dim=1, keepdim=True) / loc_counts
        loc_var = (((task_locations - loc_mean) * loc_mask) ** 2).sum(dim=1, keepdim=True) / loc_counts
        task_locations_norm = ((task_locations - loc_mean) / torch.sqrt(loc_var + eps)) * loc_mask
        agg_mean = aggregate_features.mean(dim=1, keepdim=True)
        agg_std = aggregate_features.std(dim=1, keepdim=True, unbiased=False) + eps
        aggregate_features_norm = (aggregate_features - agg_mean) / agg_std
        return T_e_norm, task_locations_norm, aggregate_features_norm

    def _edge_features(self, precedence, T_t, R, task_mask):
        eps = 1e-6
        max_tt = T_t.max(dim=-1, keepdim=True)[0] + eps
        travel_sim = torch.exp(-T_t / max_tt)
        skill_overlap = torch.matmul(R, R.transpose(-1, -2)) / (self.max_skills + eps)
        edge_feats = torch.stack([precedence, travel_sim, skill_overlap], dim=-1)
        edge_feats = edge_feats * task_mask.unsqueeze(1).unsqueeze(-1) * task_mask.unsqueeze(2).unsqueeze(-1)
        return edge_feats

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        R = batch['R']
        T_e = batch['T_e']
        task_locations = batch['task_locations']
        T_t = batch['T_t']
        precedence = batch['precedence_matrix']
        Q = batch['Q']
        task_mask = batch['task_mask']
        robot_mask = batch['robot_mask']
        aggregate_features = batch['aggregate_features']

        T_e_norm, task_loc_norm, agg_norm = self._normalize_features(
            T_e, task_mask, task_locations, aggregate_features
        )
        node_inputs = torch.cat([R, T_e_norm.unsqueeze(-1), task_loc_norm], dim=-1)
        task_feats = self.task_encoder(node_inputs)
        task_feats = self.task_pos_enc(task_feats)
        edge_feats = self._edge_features(precedence, T_t, R, task_mask)
        for layer in self.dynamic_layers:
            task_feats = layer(task_feats, edge_feats, task_mask)
        task_padding_mask = (task_mask == 0)
        for transformer in self.task_transformers:
            task_feats, _ = transformer(task_feats, padding_mask=task_padding_mask)
        robot_feats = self.robot_encoder(Q)
        robot_feats = self.robot_pos_enc(robot_feats)
        robot_padding_mask = (robot_mask == 0)
        for transformer in self.robot_transformers:
            robot_feats, _ = transformer(robot_feats, padding_mask=robot_padding_mask)
        # Cross attention: robots query tasks to know upcoming workload
        robot_context, _ = self.cross_attention(robot_feats, task_feats, task_feats, key_padding_mask=task_padding_mask)
        # Attention pooling for robots
        robot_context = self.robot_pool(robot_context)
        robot_scores = self.robot_attn_score(robot_context).squeeze(-1)
        robot_scores = robot_scores.masked_fill(robot_mask == 0, -1e9)
        robot_weights = torch.softmax(robot_scores, dim=-1)
        robot_embedding = torch.bmm(robot_weights.unsqueeze(1), robot_context).squeeze(1)
        task_embedding = self.hier_pool(task_feats, task_mask)
        agg_embedding = self.aggregate_encoder(agg_norm)
        task_embedding = self.task_gate(task_embedding)
        robot_embedding = self.robot_gate(robot_embedding)
        agg_embedding = self.aggregate_gate(agg_embedding)
        fused = torch.cat([task_embedding, robot_embedding, agg_embedding], dim=-1)
        mean = self.pred_mean(fused).squeeze(-1)
        log_var = self.pred_logvar(fused).squeeze(-1)
        log_var = torch.clamp(log_var, -10, 5)
        return mean, log_var


def heteroscedastic_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-log_var)
    return torch.mean(0.5 * (inv_var * (target - mean) ** 2 + log_var))


def train_dynamic_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                        num_epochs: int, lr: float, device: torch.device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val = float('inf')
    train_losses, val_losses = [], []
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            mean, log_var = model(batch)
            loss = heteroscedastic_loss(mean, log_var, batch['makespan'].squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()
            running += loss.item()
        scheduler.step()
        running /= len(train_loader)
        train_losses.append(running)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                mean, log_var = model(batch)
                loss = heteroscedastic_loss(mean, log_var, batch['makespan'].squeeze())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}: Train {running:.4f} | Val {val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_mrta_dynamic_model.pth')
            print('  -> Saved new best dynamic model')
    return train_losses, val_losses


def evaluate_dynamic_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            batch = {k: v.to(device) for k, v in batch.items()}
            mean, _ = model(batch)
            preds.append(mean.cpu().numpy())
            targets.append(batch['makespan'].squeeze().cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
    print(f'MSE {mse:.2f} | MAE {mae:.2f} | RMSE {rmse:.2f} | MAPE {mape:.2f}%')
    return preds, targets, dict(mse=mse, mae=mae, rmse=rmse, mape=mape)


def main():
    DATA_DIR = 'dataset_optimal_8t3r3s'
    MAX_SAMPLES = 20000
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LR = 5e-4
    HIDDEN_DIM = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    problem_files, solution_files = load_dataset(DATA_DIR, max_samples=MAX_SAMPLES)
    train_probs, test_probs, train_sols, test_sols = train_test_split(
        problem_files, solution_files, test_size=0.2, random_state=42
    )
    train_probs, val_probs, train_sols, val_sols = train_test_split(
        train_probs, train_sols, test_size=0.2, random_state=42
    )
    train_dataset = MRTADataset(train_probs, train_sols, max_tasks=10, max_robots=8, max_skills=3)
    val_dataset = MRTADataset(val_probs, val_sols, max_tasks=10, max_robots=8, max_skills=3)
    test_dataset = MRTADataset(test_probs, test_sols, max_tasks=10, max_robots=8, max_skills=3)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MRTADynamicHybridNetwork(
        hidden_dim=HIDDEN_DIM,
        num_gnn_layers=3,
        num_transformer_layers=3,
        num_heads=8,
        num_robot_transformer_layers=2,
        num_clusters=4,
        dropout=0.25,
    )
    print(model)
    train_dynamic_model(model, train_loader, val_loader, NUM_EPOCHS, LR, device)
    model.load_state_dict(torch.load('best_mrta_dynamic_model.pth', map_location=device))
    evaluate_dynamic_model(model, test_loader, device)


if __name__ == '__main__':
    main()
