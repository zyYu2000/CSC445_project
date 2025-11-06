"""
Structurally Improved MRTA Model
Focus on better feature learning and representation, not just hyperparameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mrta_deep_learning_model import MRTADataset, load_dataset, evaluate_model
from visualization_utils import plot_comprehensive_performance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class ImprovedTaskEncoder(nn.Module):
    """Better task feature encoder with attention"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        # x: (batch, num_tasks, input_dim)
        x = self.input_proj(x)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=(mask == 0) if mask is not None else None)
        x = self.norm(x + attn_out)
        return x


class ImprovedGraphEncoder(nn.Module):
    """Better graph encoder with edge features"""
    def __init__(self, node_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(node_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ))
        
    def forward(self, node_features, adj_matrix, mask):
        # node_features: (batch, num_nodes, node_dim)
        # adj_matrix: (batch, num_nodes, num_nodes)
        # mask: (batch, num_nodes)
        
        h = node_features
        for layer in self.layers:
            # Graph convolution: H' = σ(AHW)
            h_new = layer(h)
            # Apply adjacency
            h_new = torch.bmm(adj_matrix, h_new)
            # Residual connection
            h = h + h_new if h.shape == h_new.shape else h_new
            # Apply mask
            h = h * mask.unsqueeze(-1)
        return h


class StructuralMRTAModel(nn.Module):
    """Structurally improved model with better feature learning"""
    
    def __init__(self, max_tasks=10, max_robots=8, max_skills=3, hidden_dim=256):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_robots = max_robots
        self.max_skills = max_skills
        
        # ========== Better Feature Extraction ==========
        
        # Task features: [R (skills), T_e (execution), locations (x,y)]
        task_feature_dim = max_skills + 3  # R + T_e + locations
        
        # Improved task encoder with self-attention
        self.task_encoder = ImprovedTaskEncoder(task_feature_dim, hidden_dim)
        
        # Robot encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(max_skills, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Graph encoder for task relationships
        self.graph_encoder = ImprovedGraphEncoder(hidden_dim, hidden_dim, num_layers=3)
        
        # ========== Better Aggregation ==========
        
        # Attention-based pooling (learns which tasks matter)
        self.task_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Robot aggregation
        self.robot_pool = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # ========== Better Feature Interaction ==========
        
        # Project robot features to match task dimension for cross-attention
        self.robot_proj = nn.Linear(hidden_dim // 2, hidden_dim)
        
        # Cross-attention: tasks attend to robots (which robot can do which task)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        
        # ========== Better Prediction Head ==========
        
        # Aggregate features encoder
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # Final predictor with better structure
        # Input: task features + robot features + cross-attention + aggregate
        predictor_input = hidden_dim + (hidden_dim // 4) * max_robots + hidden_dim + (hidden_dim // 2)
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
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
        
        # ========== Build Task Features ==========
        T_e_expanded = T_e.unsqueeze(-1)  # (batch, max_tasks, 1)
        task_features_raw = torch.cat([R, T_e_expanded, task_locations], dim=-1)
        
        # Encode tasks with self-attention
        task_features = self.task_encoder(task_features_raw, mask=task_mask)
        
        # ========== Build Graph Structure ==========
        # Create adjacency from precedence and travel times
        # Normalize travel times
        T_t_normalized = F.softmax(-T_t / (T_t.max(dim=-1, keepdim=True)[0] + 1e-8), dim=-1)
        
        # Combine precedence (directed) and travel (symmetric, weighted)
        adj = precedence_matrix + 0.4 * T_t_normalized + 0.4 * T_t_normalized.transpose(-2, -1)
        adj = adj * task_mask.unsqueeze(1) * task_mask.unsqueeze(2)
        
        # Add self-loops and normalize
        adj = adj + torch.eye(self.max_tasks, device=adj.device).unsqueeze(0)
        degree = adj.sum(dim=-1, keepdim=True) + 1e-8
        adj = adj / degree
        
        # Graph convolution
        task_graph_features = self.graph_encoder(task_features, adj, task_mask)
        
        # ========== Robot Features ==========
        robot_features = self.robot_encoder(Q)  # (batch, max_robots, hidden_dim//2)
        robot_features = robot_features * robot_mask.unsqueeze(-1)
        
        # Project robot features to match task feature dimension for cross-attention
        robot_features_proj = self.robot_proj(robot_features)  # (batch, max_robots, hidden_dim)
        
        # ========== Cross-Attention: Tasks attend to Robots ==========
        # This learns which robots are relevant for which tasks
        cross_attn_out, _ = self.cross_attention(
            task_graph_features, robot_features_proj, robot_features_proj,
            key_padding_mask=(robot_mask == 0) if robot_mask is not None else None
        )
        cross_attn_out = cross_attn_out * task_mask.unsqueeze(-1)
        
        # ========== Task-Level Pooling ==========
        # Attention-based pooling
        task_scores = self.task_attention(task_graph_features)  # (batch, max_tasks, 1)
        task_scores = task_scores.masked_fill(task_mask.unsqueeze(-1) == 0, -1e9)
        task_weights = F.softmax(task_scores, dim=1)
        task_pooled = (task_graph_features * task_weights).sum(dim=1)  # (batch, hidden_dim)
        
        # Also pool cross-attention features
        cross_pooled = (cross_attn_out * task_weights).sum(dim=1)  # (batch, hidden_dim)
        
        # ========== Robot-Level Pooling ==========
        robot_pooled = self.robot_pool(robot_features)  # (batch, max_robots, hidden_dim//4)
        robot_pooled = robot_pooled * robot_mask.unsqueeze(-1)
        robot_flat = robot_pooled.view(batch_size, -1)  # (batch, max_robots * hidden_dim//4)
        
        # ========== Aggregate Features ==========
        aggregate_encoded = self.aggregate_encoder(aggregate_features)
        
        # ========== Combine All Features ==========
        combined = torch.cat([
            task_pooled,      # Task graph features
            robot_flat,       # Robot features
            cross_pooled,     # Cross-attention features
            aggregate_encoded # Aggregate stats
        ], dim=1)
        
        # ========== Predict ==========
        makespan = self.predictor(combined)
        
        return makespan.squeeze(-1)


def train_model_structural(model, train_loader, val_loader, num_epochs=40, lr=0.001, device='cuda'):
    """Training with adaptive learning rate and better strategies"""
    criterion = nn.MSELoss()
    
    # Use AdamW with better weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.999))
    
    # Reduce learning rate on plateau - more adaptive than step decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, 
        min_lr=1e-6, threshold=0.01
    )
    
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15  # Early stopping
    
    model.to(device)
    
    # Initialize model weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.fill_(0)
            m.weight.data.fill_(1.0)
    
    model.apply(init_weights)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            makespan_pred = model(batch)
            loss = criterion(makespan_pred, batch['makespan'].squeeze())
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                makespan_pred = model(batch)
                loss = criterion(makespan_pred, batch['makespan'].squeeze())
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_structural_model.pth')
            print(f'  -> New best model saved! (Val Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'\nEarly stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)')
                break
        
        # If learning rate gets too small, might be stuck
        if current_lr < 1e-6:
            print(f'\nLearning rate too small ({current_lr:.2e}), stopping training')
            break
    
    return train_losses, val_losses, learning_rates


def main():
    """Main training script"""
    DATA_DIR = 'dataset_optimal_8t3r3s'
    MAX_SAMPLES = 2000
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 256
    MAX_TASKS = 10
    MAX_ROBOTS = 8
    MAX_SKILLS = 3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('=' * 60)
    print('STRUCTURALLY IMPROVED MRTA Model')
    print('Key Improvements:')
    print('  - Better task encoding with self-attention')
    print('  - Improved graph encoder with residual connections')
    print('  - Cross-attention between tasks and robots')
    print('  - Attention-based pooling')
    print('  - Simpler, more stable training')
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
    
    # Create model
    model = StructuralMRTAModel(
        max_tasks=MAX_TASKS,
        max_robots=MAX_ROBOTS,
        max_skills=MAX_SKILLS,
        hidden_dim=HIDDEN_DIM
    )
    
    print(f'\nModel parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Train model
    print('\n' + '=' * 60)
    print('Starting training...')
    print('=' * 60)
    train_losses, val_losses, learning_rates = train_model_structural(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 3, 3)
    # Plot validation loss improvement
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss (Zoom)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('structural_training_curves.png', dpi=150)
    print('\nSaved training curves to structural_training_curves.png')
    
    # Load best model and evaluate
    print('\nLoading best model for evaluation...')
    model.load_state_dict(torch.load('best_structural_model.pth', weights_only=True))
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    predictions, targets, metrics = evaluate_model(model, test_loader, device=device)
    
    # Create comprehensive performance visualization
    print('\nGenerating performance visualization...')
    plot_comprehensive_performance(
        predictions, targets, metrics, 
        model_name="Structurally Improved MRTA",
        save_path="structural_model_performance.png"
    )
    
    print('\n' + '=' * 60)
    print('Training completed!')
    print('=' * 60)


if __name__ == '__main__':
    import torch.optim as optim
    main()

