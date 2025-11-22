"""
Hybrid GNN + Transformer Model for Multi-Robot Task Allocation (MRTA) Makespan Prediction

This model combines:
- GNN: Captures local task relationships (precedence, travel times)
- Transformer: Models global task ordering and long-range dependencies
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import glob
from tqdm import tqdm
import math
try:
    from scipy import stats
except ImportError:
    print("Warning: scipy not available. Q-Q plot will be skipped.")
    stats = None


class SimpleGCNLayer(nn.Module):
    """Simple Graph Convolutional Layer"""
    def __init__(self, in_features, out_features):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: (batch, num_nodes, in_features)
        # adj: (batch, num_nodes, num_nodes) - adjacency matrix
        support = self.linear(x)
        output = torch.bmm(adj, support)
        return F.relu(output)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention for Transformer"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: (batch, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask should be (batch, seq_len, seq_len) or (batch, 1, seq_len, seq_len)
            # Expand to match scores shape (batch, num_heads, seq_len, seq_len)
            if mask.dim() == 3:
                # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            # Now mask is (batch, 1, seq_len, seq_len) and will broadcast to (batch, num_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(attn_output)
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with Self-Attention"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class MRTAGNNTransformerNetwork(nn.Module):
    """Hybrid GNN + Transformer model for MRTA makespan prediction"""
    
    def __init__(self, max_tasks=10, max_robots=8, max_skills=3, 
                 hidden_dim=256, num_gnn_layers=2, num_transformer_layers=2,
                 num_heads=8, d_ff=512, dropout=0.2):
        super(MRTAGNNTransformerNetwork, self).__init__()
        
        self.max_tasks = max_tasks
        self.max_robots = max_robots
        self.max_skills = max_skills
        self.hidden_dim = hidden_dim
        
        # ========== GNN Branch: Local Task Relationships ==========
        
        # Node features: [R (skills), T_e (execution), locations (x,y)]
        node_feature_dim = max_skills + 3
        
        # Initial node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN layers for local relationships
        self.gnn_layers = nn.ModuleList([
            SimpleGCNLayer(hidden_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_gnn_layers)
        ])
        
        # ========== Transformer Branch: Global Task Ordering ==========
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_tasks)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, d_ff, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # ========== Robot Features ==========
        
        self.robot_encoder = nn.Sequential(
            nn.Linear(max_skills, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # ========== Cross-Modal Fusion ==========
        
        # Fusion layer to combine GNN and Transformer outputs
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # GNN + Transformer
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ========== Graph-Level Pooling ==========
        
        # Attention-based pooling for tasks
        self.task_pool_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # ========== Aggregate Features ==========
        
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ========== Final Predictor ==========
        
        # Input: fused graph embedding + robot features + aggregate features
        predictor_input_dim = hidden_dim + (hidden_dim // 4) * max_robots + (hidden_dim // 2)
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Static identity matrix for masking/self-loops
        self.register_buffer('task_eye', torch.eye(max_tasks))
    
    def forward(self, x):
        R = x['R']  # (batch, max_tasks, max_skills)
        T_e = x['T_e']  # (batch, max_tasks)
        task_locations = x['task_locations']  # (batch, max_tasks, 2)
        T_t = x['T_t']  # (batch, max_tasks, max_tasks)
        precedence_matrix = x['precedence_matrix']  # (batch, max_tasks, max_tasks)
        Q = x['Q']  # (batch, max_robots, max_skills)
        task_mask = x['task_mask']  # (batch, max_tasks)
        robot_mask = x['robot_mask']  # (batch, max_robots)
        aggregate_features = x['aggregate_features']  # (batch, 16)
        
        batch_size = R.size(0)
        eps = 1e-6
        
        # ========== Feature Normalization ==========
        
        # Normalize execution times with task mask support
        task_counts = task_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        te_mean = (T_e * task_mask).sum(dim=1, keepdim=True) / task_counts
        te_var = (((T_e - te_mean) * task_mask) ** 2).sum(dim=1, keepdim=True) / task_counts
        T_e_norm = ((T_e - te_mean) / torch.sqrt(te_var + eps)) * task_mask
        
        # Normalize task coordinates
        loc_mask = task_mask.unsqueeze(-1)
        loc_counts = loc_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        loc_mean = (task_locations * loc_mask).sum(dim=1, keepdim=True) / loc_counts
        loc_var = (((task_locations - loc_mean) * loc_mask) ** 2).sum(dim=1, keepdim=True) / loc_counts
        task_locations_norm = ((task_locations - loc_mean) / torch.sqrt(loc_var + eps)) * loc_mask
        
        # Normalize aggregate features (per sample)
        agg_mean = aggregate_features.mean(dim=1, keepdim=True)
        agg_std = aggregate_features.std(dim=1, keepdim=True, unbiased=False) + eps
        aggregate_features_norm = (aggregate_features - agg_mean) / agg_std
        
        # ========== Build Node Features ==========
        T_e_expanded = T_e_norm.unsqueeze(-1)  # (batch, max_tasks, 1)
        node_features = torch.cat([R, T_e_expanded, task_locations_norm], dim=-1)  # (batch, max_tasks, max_skills+3)
        
        # Initial node embeddings
        node_embeddings = self.node_encoder(node_features)  # (batch, max_tasks, hidden_dim)
        
        # ========== GNN Branch: Local Relationships ==========
        
        # Build adjacency matrix from precedence and travel times
        # Precedence creates directed edges
        # Travel times create weighted edges
        max_tt = T_t.max(dim=-1, keepdim=True)[0] + 1e-8
        tt_scores = -T_t / max_tt
        very_negative = -1e9
        diag_mask = self.task_eye.unsqueeze(0).bool()
        tt_scores = tt_scores.masked_fill(diag_mask, very_negative)
        # Mask invalid tasks so padding rows/cols don't leak mass
        tt_scores = tt_scores.masked_fill(task_mask.unsqueeze(-1) == 0, very_negative)
        tt_scores = tt_scores.masked_fill(task_mask.unsqueeze(1) == 0, very_negative)
        T_t_normalized = F.softmax(tt_scores, dim=-1)
        T_t_normalized = T_t_normalized * task_mask.unsqueeze(-1)
        
        # Combine precedence (directed) and travel (symmetric)
        adj = precedence_matrix + 0.3 * T_t_normalized + 0.3 * T_t_normalized.transpose(-2, -1)
        adj = adj * task_mask.unsqueeze(1) * task_mask.unsqueeze(2)
        
        # Normalize with self-loops
        adj = adj + torch.eye(self.max_tasks, device=adj.device).unsqueeze(0)
        degree = adj.sum(dim=-1, keepdim=True) + 1e-8
        adj = adj / degree
        
        # Apply GNN layers
        gnn_output = node_embeddings
        for gnn_layer in self.gnn_layers:
            gnn_output = gnn_layer(gnn_output, adj)
            gnn_output = gnn_output * task_mask.unsqueeze(-1)
        
        # ========== Transformer Branch: Global Ordering ==========
        
        # Add positional encoding
        transformer_input = self.pos_encoder(node_embeddings)
        
        # Create attention mask from task mask
        # mask: (batch, max_tasks, max_tasks) - 1 where valid, 0 where invalid
        attention_mask = task_mask.unsqueeze(1) * task_mask.unsqueeze(2)  # (batch, max_tasks, max_tasks)
        
        # Apply transformer layers
        transformer_output = transformer_input
        all_attn_weights = []
        for transformer_layer in self.transformer_layers:
            transformer_output, attn_weights = transformer_layer(transformer_output, attention_mask)
            transformer_output = transformer_output * task_mask.unsqueeze(-1)
            all_attn_weights.append(attn_weights)
        
        # ========== Fusion: Combine GNN and Transformer ==========
        
        # Concatenate GNN and Transformer outputs
        fused_features = torch.cat([gnn_output, transformer_output], dim=-1)  # (batch, max_tasks, hidden_dim*2)
        fused_features = self.fusion_layer(fused_features)  # (batch, max_tasks, hidden_dim)
        fused_features = fused_features * task_mask.unsqueeze(-1)
        
        # ========== Graph-Level Pooling with Attention ==========
        
        # Attention-based pooling (learns which tasks are most important)
        attention_scores = self.task_pool_attention(fused_features)  # (batch, max_tasks, 1)
        attention_scores = attention_scores.masked_fill(task_mask.unsqueeze(-1) == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, max_tasks, 1)
        
        # Weighted sum
        graph_embedding = (fused_features * attention_weights).sum(dim=1)  # (batch, hidden_dim)
        
        # ========== Robot Features ==========
        
        robot_embeddings = self.robot_encoder(Q)  # (batch, max_robots, hidden_dim//4)
        robot_embeddings = robot_embeddings * robot_mask.unsqueeze(-1)
        robot_flat = robot_embeddings.view(batch_size, -1)  # (batch, max_robots * hidden_dim//4)
        
        # ========== Aggregate Features ==========
        
        aggregate_encoded = self.aggregate_encoder(aggregate_features_norm)  # (batch, hidden_dim//2)
        
        # ========== Final Prediction ==========
        
        # Concatenate all features
        combined = torch.cat([graph_embedding, robot_flat, aggregate_encoded], dim=1)
        
        # Predict makespan
        makespan_pred = self.predictor(combined)
        
        return makespan_pred.squeeze(-1)


# Import dataset and training functions from original model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mrta_deep_learning_model import MRTADataset, load_dataset, train_model, evaluate_model, plot_training_curves


def plot_comprehensive_performance(predictions, targets, metrics, model_name="Model", save_path="performance_analysis.png"):
    """Create comprehensive performance visualization"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Calculate additional metrics
    errors = predictions - targets
    abs_errors = np.abs(errors)
    relative_errors = (errors / (targets + 1e-8)) * 100
    
    # 1. Predictions vs Targets (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(targets, predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    ax1.plot(targets, p(targets), "g--", alpha=0.8, lw=2, 
             label=f'Regression (slope={z[0]:.3f})')
    
    ax1.set_xlabel('True Makespan', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Makespan', fontsize=11, fontweight='bold')
    ax1.set_title(f'{model_name}: Predictions vs True Values\nMAPE: {metrics["mape"]:.2f}%', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error Distribution (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors):.2f}')
    ax2.set_xlabel('Prediction Error (Predicted - True)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Absolute Error Distribution (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(abs_errors, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    ax3.axvline(x=metrics['mae'], color='r', linestyle='--', linewidth=2, 
                label=f'MAE: {metrics["mae"]:.2f}')
    ax3.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Residual Plot (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(targets, errors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('True Makespan', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Residuals (Predicted - True)', fontsize=11, fontweight='bold')
    ax4.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Relative Error Distribution (Middle Middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(relative_errors, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.axvline(x=metrics['mape'], color='orange', linestyle='--', linewidth=2, 
                label=f'MAPE: {metrics["mape"]:.2f}%')
    ax5.axvline(x=-metrics['mape'], color='orange', linestyle='--', linewidth=2)
    ax5.set_xlabel('Relative Error (%)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Relative Error Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Q-Q Plot (Middle Right)
    ax6 = fig.add_subplot(gs[1, 2])
    if stats is not None:
        stats.probplot(errors, dist="norm", plot=ax6)
        ax6.set_title('Q-Q Plot (Error Normality Check)', fontsize=12, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Q-Q Plot\n(scipy not available)', 
                ha='center', va='center', fontsize=12)
        ax6.set_title('Q-Q Plot (Not Available)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Metrics Bar Chart (Bottom Left)
    ax7 = fig.add_subplot(gs[2, 0])
    metric_names = ['MAE', 'RMSE', 'MAPE (%)']
    metric_values = [metrics['mae'], metrics['rmse'], metrics['mape']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax7.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax7.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax7.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 8. Error by Makespan Range (Bottom Middle)
    ax8 = fig.add_subplot(gs[2, 1])
    # Divide into bins
    bins = np.linspace(targets.min(), targets.max(), 6)
    bin_indices = np.digitize(targets, bins)
    bin_errors = [abs_errors[bin_indices == i].mean() for i in range(1, len(bins))]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    ax8.bar(bin_centers, bin_errors, width=(bins[1]-bins[0])*0.8, 
            alpha=0.7, color='coral', edgecolor='black', linewidth=1.5)
    ax8.set_xlabel('Makespan Range', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax8.set_title('Error by Makespan Range', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Summary Statistics Table (Bottom Right)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate additional statistics
    correlation = np.corrcoef(targets, predictions)[0, 1]
    r2 = 1 - (np.sum((targets - predictions)**2) / np.sum((targets - np.mean(targets))**2))
    bias = np.mean(errors)
    std_error = np.std(errors)
    
    stats_text = f"""
    Performance Summary
    
    Metrics:
    • MAE:  {metrics['mae']:.2f}
    • RMSE: {metrics['rmse']:.2f}
    • MAPE: {metrics['mape']:.2f}%
    
    Statistics:
    • Correlation: {correlation:.4f}
    • R² Score:    {r2:.4f}
    • Bias:        {bias:.2f}
    • Std Error:   {std_error:.2f}
    
    Data:
    • Samples:     {len(targets)}
    • Min Error:   {errors.min():.2f}
    • Max Error:   {errors.max():.2f}
    """
    
    ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{model_name} - Comprehensive Performance Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\nSaved comprehensive performance analysis to {save_path}')
    
    return fig


def main():
    """Main training script for Hybrid GNN + Transformer model"""
    # Configuration
    DATA_DIR = 'dataset_optimal_8t3r3s'
    MAX_SAMPLES = None  # Use full dataset for better generalization
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 256
    NUM_GNN_LAYERS = 2
    NUM_TRANSFORMER_LAYERS = 2
    NUM_HEADS = 8
    D_FF = 512
    DROPOUT = 0.3
    MAX_TASKS = 10
    MAX_ROBOTS = 8
    MAX_SKILLS = 3
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('=' * 60)
    print('Hybrid GNN + Transformer Model for MRTA')
    print('=' * 60)
    
    # Load dataset
    print('\nLoading dataset...')
    problem_files, solution_files = load_dataset(DATA_DIR, max_samples=MAX_SAMPLES)
    
    if len(problem_files) == 0:
        print("No matched problem-solution pairs found!")
        return
    
    # Split dataset
    train_probs, test_probs, train_sols, test_sols = train_test_split(
        problem_files, solution_files, test_size=0.2, random_state=42
    )
    train_probs, val_probs, train_sols, val_sols = train_test_split(
        train_probs, train_sols, test_size=0.2, random_state=42
    )
    
    print(f'Train: {len(train_probs)}, Val: {len(val_probs)}, Test: {len(test_probs)}')
    
    # Create datasets
    train_dataset = MRTADataset(train_probs, train_sols, MAX_TASKS, MAX_ROBOTS, MAX_SKILLS)
    val_dataset = MRTADataset(val_probs, val_sols, MAX_TASKS, MAX_ROBOTS, MAX_SKILLS)
    test_dataset = MRTADataset(test_probs, test_sols, MAX_TASKS, MAX_ROBOTS, MAX_SKILLS)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create Hybrid model
    model = MRTAGNNTransformerNetwork(
        max_tasks=MAX_TASKS,
        max_robots=MAX_ROBOTS,
        max_skills=MAX_SKILLS,
        hidden_dim=HIDDEN_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    )
    
    print(f'\nHybrid Model Architecture:')
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total_params:,}')
    print(f'GNN layers: {NUM_GNN_LAYERS}')
    print(f'Transformer layers: {NUM_TRANSFORMER_LAYERS}')
    print(f'Attention heads: {NUM_HEADS}')
    
    # Train model
    print('\n' + '=' * 60)
    print('Starting training...')
    print('=' * 60)
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Load best model and evaluate
    print('\nLoading best model for evaluation...')
    model.load_state_dict(torch.load('best_mrta_model.pth', weights_only=True))
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    predictions, targets, metrics = evaluate_model(model, test_loader, device=device)
    
    # Create comprehensive performance visualization
    print('\nGenerating performance visualization...')
    plot_comprehensive_performance(
        predictions, targets, metrics, 
        model_name="Hybrid GNN + Transformer",
        save_path="hybrid_model_performance.png"
    )
    
    print('\n' + '=' * 60)
    print('Training completed!')
    print('=' * 60)


if __name__ == '__main__':
    main()
