"""
Graph Neural Network (GNN) Model for Multi-Robot Task Allocation (MRTA) Makespan Prediction

This script implements a GNN-based model that treats tasks as nodes and uses
precedence constraints and travel times to form edges in a graph structure.
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

# Try to import PyTorch Geometric, fallback to manual implementation if not available
try:
    from torch_geometric.nn import GCNConv, GATConv, MessagePassing, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("PyTorch Geometric not available. Using manual GNN implementation.")


class SimpleGCNLayer(nn.Module):
    """Simple Graph Convolutional Layer (manual implementation)"""
    def __init__(self, in_features, out_features):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: (batch, num_nodes, in_features)
        # adj: (batch, num_nodes, num_nodes) - adjacency matrix
        # Simple GCN: H' = σ(AHW)
        support = self.linear(x)  # (batch, num_nodes, out_features)
        output = torch.bmm(adj, support)  # Batch matrix multiplication
        return F.relu(output)


class MRTAGNNNetwork(nn.Module):
    """GNN-based model for MRTA makespan prediction"""
    
    def __init__(self, max_tasks=10, max_robots=8, max_skills=3, 
                 hidden_dim=128, num_gnn_layers=3, dropout=0.2, use_pyg=False):
        super(MRTAGNNNetwork, self).__init__()
        
        self.max_tasks = max_tasks
        self.max_robots = max_robots
        self.max_skills = max_skills
        self.use_pyg = use_pyg and PYG_AVAILABLE
        
        # Node features for tasks:
        # - Task requirements (R): max_skills dims
        # - Execution time (T_e): 1 dim
        # - Location (x, y): 2 dims
        # Total: max_skills + 3 dims
        node_feature_dim = max_skills + 3
        
        # Initial node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Robot encoder (for robot features)
        self.robot_encoder = nn.Sequential(
            nn.Linear(max_skills, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN layers
        if self.use_pyg:
            # Use PyTorch Geometric
            self.gnn_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=True) 
                if i == 0 else
                GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout, concat=(i < num_gnn_layers - 1))
                for i in range(num_gnn_layers)
            ])
            self.gnn_output_dim = hidden_dim * 4 if num_gnn_layers > 1 else hidden_dim
        else:
            # Manual GNN implementation
            self.gnn_layers = nn.ModuleList([
                SimpleGCNLayer(hidden_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_gnn_layers)
            ])
            self.gnn_output_dim = hidden_dim
        
        # Graph-level pooling
        self.pool = nn.Sequential(
            nn.Linear(self.gnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Robot features aggregation
        self.robot_pool = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Aggregate features encoder
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final predictor
        # Input: graph embedding + robot features + aggregate features
        predictor_input_dim = hidden_dim + (hidden_dim // 4) * max_robots + (hidden_dim // 2)
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        if self.use_pyg:
            return self.forward_pyg(x)
        else:
            return self.forward_manual(x)
    
    def forward_manual(self, x):
        """Manual GNN forward pass"""
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
        
        # Build node features: [R, T_e, locations]
        T_e_expanded = T_e.unsqueeze(-1)  # (batch, max_tasks, 1)
        node_features = torch.cat([R, T_e_expanded, task_locations], dim=-1)  # (batch, max_tasks, max_skills+3)
        
        # Encode nodes
        node_embeddings = self.node_encoder(node_features)  # (batch, max_tasks, hidden_dim)
        
        # Build adjacency matrix from travel times and precedence
        # Combine precedence constraints and travel time relationships
        # Normalize travel time matrix to create edge weights
        T_t_normalized = F.softmax(-T_t / (T_t.max(dim=-1, keepdim=True)[0] + 1e-8), dim=-1)
        adj = precedence_matrix + 0.5 * T_t_normalized  # Combine precedence and travel relationships
        adj = adj * task_mask.unsqueeze(1) * task_mask.unsqueeze(2)  # Apply masks
        
        # Normalize adjacency matrix
        adj = adj + torch.eye(self.max_tasks, device=adj.device).unsqueeze(0)  # Add self-loops
        degree = adj.sum(dim=-1, keepdim=True) + 1e-8
        adj = adj / degree  # Normalize
        
        # Apply GNN layers
        h = node_embeddings
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, adj)
            h = h * task_mask.unsqueeze(-1)  # Apply mask
        
        # Graph-level pooling (mean over tasks)
        graph_embedding = (h * task_mask.unsqueeze(-1)).sum(dim=1) / (task_mask.sum(dim=1, keepdim=True) + 1e-8)
        graph_embedding = self.pool(graph_embedding)  # (batch, hidden_dim)
        
        # Encode robot features
        robot_embeddings = self.robot_encoder(Q)  # (batch, max_robots, hidden_dim//2)
        robot_embeddings = robot_embeddings * robot_mask.unsqueeze(-1)
        robot_flat = self.robot_pool(robot_embeddings).view(batch_size, -1)  # (batch, max_robots * hidden_dim//4)
        
        # Encode aggregate features
        aggregate_encoded = self.aggregate_encoder(aggregate_features)  # (batch, hidden_dim//2)
        
        # Concatenate all features
        combined = torch.cat([graph_embedding, robot_flat, aggregate_encoded], dim=1)
        
        # Predict makespan
        makespan_pred = self.predictor(combined)
        
        return makespan_pred.squeeze(-1)
    
    def forward_pyg(self, x):
        """PyTorch Geometric forward pass (if available)"""
        # This would use PyTorch Geometric's Data and Batch objects
        # For now, fall back to manual implementation
        return self.forward_manual(x)


# Reuse the same dataset class from the original model
# (We'll import it or copy the necessary parts)
def load_dataset(data_dir: str, max_samples: int = None) -> Tuple[List[str], List[str]]:
    """Load problem and solution file paths"""
    problem_dir = os.path.join(data_dir, 'problem_instances')
    solution_dir = os.path.join(data_dir, 'solutions')
    
    problem_files = sorted(glob.glob(os.path.join(problem_dir, '*.json')))
    
    problem_files_matched = []
    solution_files_matched = []
    
    for prob_file in problem_files:
        basename = os.path.basename(prob_file)
        parts = basename.replace('.json', '').split('_')
        
        if len(parts) >= 3:
            idx = parts[-1]
            prefix = parts[-2]
            
            sol_pattern = os.path.join(solution_dir, f'optimal_schedule_{prefix}_{idx}.json')
            if os.path.exists(sol_pattern):
                problem_files_matched.append(prob_file)
                solution_files_matched.append(sol_pattern)
            else:
                sol_pattern2 = os.path.join(solution_dir, f'optimal_schedule_*_{idx}.json')
                matches = glob.glob(sol_pattern2)
                if matches:
                    problem_files_matched.append(prob_file)
                    solution_files_matched.append(matches[0])
    
    if max_samples:
        problem_files_matched = problem_files_matched[:max_samples]
        solution_files_matched = solution_files_matched[:max_samples]
    
    print(f"Found {len(problem_files_matched)} matched problem-solution pairs")
    return problem_files_matched, solution_files_matched


# Import the dataset class from the original file
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mrta_deep_learning_model import MRTADataset, train_model, evaluate_model, plot_training_curves


def main():
    """Main training script for GNN model"""
    # Configuration
    DATA_DIR = 'dataset_optimal_8t3r3s'
    MAX_SAMPLES = 5000
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 256
    NUM_GNN_LAYERS = 3
    DROPOUT = 0.3
    MAX_TASKS = 10
    MAX_ROBOTS = 8
    MAX_SKILLS = 3
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'PyTorch Geometric available: {PYG_AVAILABLE}')
    
    # Load dataset
    print('Loading dataset...')
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
    
    # Create GNN model
    model = MRTAGNNNetwork(
        max_tasks=MAX_TASKS,
        max_robots=MAX_ROBOTS,
        max_skills=MAX_SKILLS,
        hidden_dim=HIDDEN_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        dropout=DROPOUT,
        use_pyg=False  # Set to True if PyTorch Geometric is installed
    )
    
    print(f'\nGNN Model architecture:')
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Train model
    print('\nStarting training...')
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
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()

