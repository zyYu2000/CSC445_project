"""
Improved Hybrid GNN + Transformer Model with Better Loss Function and Normalization
Addresses the bias and low R² issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mrta_hybrid_gnn_transformer import (
    MRTAGNNTransformerNetwork, MRTADataset, load_dataset, 
    plot_training_curves, plot_comprehensive_performance
)
from mrta_deep_learning_model import evaluate_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class QuantileLoss(nn.Module):
    """Quantile loss to reduce bias - penalizes over/under prediction differently"""
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        # For now, use median (0.5 quantile) as main prediction
        # But train with multiple quantiles to reduce bias
        if isinstance(preds, tuple):
            preds = preds[0]  # Use main prediction
        
        errors = target - preds
        loss = torch.max(
            self.quantiles[1] * errors,
            (self.quantiles[1] - 1) * errors
        )
        return loss.mean()


class HuberQuantileLoss(nn.Module):
    """Combination of Huber loss and quantile loss for robust, unbiased predictions"""
    def __init__(self, delta=1.0, quantile=0.5):
        super(HuberQuantileLoss, self).__init__()
        self.delta = delta
        self.quantile = quantile
        
    def forward(self, preds, target):
        errors = target - preds
        abs_errors = torch.abs(errors)
        
        # Quantile component
        quantile_loss = torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        
        # Huber component (less sensitive to outliers)
        huber_condition = abs_errors < self.delta
        huber_loss = torch.where(
            huber_condition,
            0.5 * errors ** 2,
            self.delta * abs_errors - 0.5 * self.delta ** 2
        )
        
        # Combine (weighted)
        return 0.7 * quantile_loss.mean() + 0.3 * huber_loss.mean()


def train_model_improved(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    """Improved training with better loss function"""
    # Use Huber-Quantile loss to reduce bias
    criterion = HuberQuantileLoss(delta=50.0, quantile=0.5)
    
    # Also track MSE for comparison
    mse_criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Use Cosine Annealing with Warm Restarts for better convergence
    # This helps escape local minima and provides smooth decay
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Alternative: StepLR with more aggressive decay
    # Uncomment if cosine annealing doesn't work well
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=6, gamma=0.7
    # )
    
    # Track learning rates
    learning_rates = []
    
    train_losses = []
    val_losses = []
    train_mses = []
    val_mses = []
    best_val_loss = float('inf')
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            makespan_pred = model(batch)
            
            # Main loss (Huber-Quantile)
            loss = criterion(makespan_pred, batch['makespan'].squeeze())
            
            # Also compute MSE for monitoring
            mse = mse_criterion(makespan_pred, batch['makespan'].squeeze())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse.item()
        
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_losses.append(train_loss)
        train_mses.append(train_mse)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                makespan_pred = model(batch)
                loss = criterion(makespan_pred, batch['makespan'].squeeze())
                mse = mse_criterion(makespan_pred, batch['makespan'].squeeze())
                val_loss += loss.item()
                val_mse += mse.item()
        
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f} (MSE: {train_mse:.2f}), '
              f'Val Loss = {val_loss:.4f} (MSE: {val_mse:.2f}), LR = {current_lr:.6f}')
        
        # Save best model based on validation MSE (for fair comparison)
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), 'best_mrta_model_improved.pth')
            print(f'  -> New best model saved! (Val MSE: {val_mse:.2f})')
    
    return train_losses, val_losses, train_mses, val_mses, learning_rates


def normalize_features(dataset):
    """Normalize makespan targets for better training"""
    makespans = dataset.makespans
    mean = makespans.mean()
    std = makespans.std()
    
    # Store normalization params
    dataset.makespan_mean = mean
    dataset.makespan_std = std
    
    # Normalize
    dataset.makespans = (makespans - mean) / (std + 1e-8)
    
    return mean, std


def denormalize_predictions(predictions, mean, std):
    """Denormalize predictions back to original scale"""
    return predictions * std + mean


def main():
    """Main training script with improvements"""
    # Configuration - Use more data
    DATA_DIR = 'dataset_optimal_8t3r3s'
    MAX_SAMPLES = 2000  # Increased from 500
    BATCH_SIZE = 32
    NUM_EPOCHS = 30 
    LEARNING_RATE = 0.001  # Slightly lower for stability
    HIDDEN_DIM = 256
    NUM_GNN_LAYERS = 2
    NUM_TRANSFORMER_LAYERS = 2
    NUM_HEADS = 8
    D_FF = 512
    DROPOUT = 0.25  # Slightly less dropout
    MAX_TASKS = 10
    MAX_ROBOTS = 8
    MAX_SKILLS = 3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('=' * 60)
    print('IMPROVED Hybrid GNN + Transformer Model')
    print('Improvements:')
    print('  - Huber-Quantile Loss (reduces bias)')
    print('  - More training data (2000 samples)')
    print('  - Better hyperparameters')
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
    
    # Normalize targets (optional - can help with training)
    # Note: We'll denormalize for evaluation
    train_mean = train_dataset.makespans.mean()
    train_std = train_dataset.makespans.std()
    
    print(f'\nTarget statistics:')
    print(f'  Mean: {train_mean:.2f}, Std: {train_std:.2f}')
    print(f'  Range: [{train_dataset.makespans.min():.2f}, {train_dataset.makespans.max():.2f}]')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
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
    
    print(f'\nModel parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Train model
    print('\n' + '=' * 60)
    print('Starting training with improved loss function...')
    print('=' * 60)
    train_losses, val_losses, train_mses, val_mses, learning_rates = train_model_improved(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device
    )
    
    # Plot training curves (loss, MSE, and learning rate)
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss (Huber-Quantile)')
    plt.plot(val_losses, label='Val Loss (Huber-Quantile)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Huber-Quantile)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_mses, label='Train MSE')
    plt.plot(val_mses, label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training MSE (for comparison)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('improved_training_curves.png', dpi=150)
    print('\nSaved training curves to improved_training_curves.png')
    
    # Load best model and evaluate
    print('\nLoading best model for evaluation...')
    model.load_state_dict(torch.load('best_mrta_model_improved.pth', weights_only=True))
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    predictions, targets, metrics = evaluate_model(model, test_loader, device=device)
    
    # Create comprehensive performance visualization
    print('\nGenerating performance visualization...')
    plot_comprehensive_performance(
        predictions, targets, metrics, 
        model_name="Improved Hybrid GNN + Transformer",
        save_path="improved_model_performance.png"
    )
    
    print('\n' + '=' * 60)
    print('Training completed!')
    print('=' * 60)
    print('\nKey Improvements Made:')
    print('  1. Huber-Quantile Loss - reduces bias')
    print('  2. More training data (2000 vs 500)')
    print('  3. Better hyperparameters')
    print('  4. Improved regularization')


if __name__ == '__main__':
    main()

