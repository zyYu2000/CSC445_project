"""
Deep Learning Model for Multi-Robot Task Allocation (MRTA) Makespan Prediction

This script implements a neural network to predict the optimal makespan for MRTA problems
using the dataset structure:
- Q: Robot skill matrix (N x S)
- R: Task requirement matrix ((M+2) x S)
- T_e: Execution times (M+2)
- T_t: Travel time matrix ((M+2) x (M+2))
- task_locations: Spatial coordinates (M+2 x 2)
- precedence_constraints: List of [i, j] pairs
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import glob
from tqdm import tqdm


class MRTADataset(Dataset):
    """Dataset class for MRTA problem instances"""
    
    def __init__(self, problem_files: List[str], solution_files: List[str], 
                 max_tasks: int = 10, max_robots: int = 8, max_skills: int = 3):
        """
        Args:
            problem_files: List of paths to problem JSON files
            solution_files: List of paths to solution JSON files (matched with problems)
            max_tasks: Maximum number of tasks (for padding)
            max_robots: Maximum number of robots (for padding)
            max_skills: Maximum number of skills (for padding)
        """
        self.problem_files = problem_files
        self.solution_files = solution_files
        self.max_tasks = max_tasks
        self.max_robots = max_robots
        self.max_skills = max_skills
        
        # Load all data
        self.data = []
        self.makespans = []
        
        print("Loading dataset...")
        for prob_file, sol_file in tqdm(zip(problem_files, solution_files), 
                                        total=len(problem_files)):
            try:
                with open(prob_file, 'r') as f:
                    problem = json.load(f)
                with open(sol_file, 'r') as f:
                    solution = json.load(f)
                
                # Extract makespan
                makespan = solution.get('makespan', 0.0)
                
                # Process and pad features
                features = self._process_problem(problem)
                self.data.append(features)
                self.makespans.append(makespan)
            except Exception as e:
                print(f"Error loading {prob_file}: {e}")
                continue
        
        self.makespans = np.array(self.makespans)
        print(f"Loaded {len(self.data)} instances")
        print(f"Makespan range: [{self.makespans.min():.2f}, {self.makespans.max():.2f}]")
        print(f"Makespan mean: {self.makespans.mean():.2f}, std: {self.makespans.std():.2f}")
    
    def _process_problem(self, problem: Dict) -> Dict:
        """Process a problem instance into fixed-size feature tensors"""
        Q = np.array(problem['Q'], dtype=np.float32)  # (N, S)
        R = np.array(problem['R'], dtype=np.float32)  # (M+2, S)
        T_e = np.array(problem['T_e'], dtype=np.float32)  # (M+2,)
        T_t = np.array(problem['T_t'], dtype=np.float32)  # (M+2, M+2)
        task_locations = np.array(problem['task_locations'], dtype=np.float32)  # (M+2, 2)
        precedence = problem.get('precedence_constraints', [])
        
        n_robots, n_skills = Q.shape
        n_tasks = R.shape[0]
        
        # Pad matrices to fixed size
        Q_padded = np.zeros((self.max_robots, self.max_skills), dtype=np.float32)
        Q_padded[:n_robots, :n_skills] = Q
        
        R_padded = np.zeros((self.max_tasks, self.max_skills), dtype=np.float32)
        R_padded[:n_tasks, :n_skills] = R
        
        T_e_padded = np.zeros(self.max_tasks, dtype=np.float32)
        T_e_padded[:n_tasks] = T_e
        
        T_t_padded = np.zeros((self.max_tasks, self.max_tasks), dtype=np.float32)
        T_t_padded[:n_tasks, :n_tasks] = T_t
        
        task_locations_padded = np.zeros((self.max_tasks, 2), dtype=np.float32)
        task_locations_padded[:n_tasks] = task_locations
        
        # Create precedence constraint matrix
        precedence_matrix = np.zeros((self.max_tasks, self.max_tasks), dtype=np.float32)
        for i, j in precedence:
            if i < self.max_tasks and j < self.max_tasks:
                precedence_matrix[i, j] = 1.0
        
        # Create mask for valid tasks
        task_mask = np.zeros(self.max_tasks, dtype=np.float32)
        task_mask[:n_tasks] = 1.0
        
        # Create mask for valid robots
        robot_mask = np.zeros(self.max_robots, dtype=np.float32)
        robot_mask[:n_robots] = 1.0
        
        # Extract additional features
        # Task statistics
        task_exec_mean = T_e[:n_tasks].mean() if n_tasks > 0 else 0.0
        task_exec_std = T_e[:n_tasks].std() if n_tasks > 0 else 0.0
        task_exec_max = T_e[:n_tasks].max() if n_tasks > 0 else 0.0
        task_exec_sum = T_e[:n_tasks].sum() if n_tasks > 0 else 0.0
        
        # Travel time statistics
        travel_times = T_t[:n_tasks, :n_tasks]
        travel_mean = travel_times[travel_times > 0].mean() if np.any(travel_times > 0) else 0.0
        travel_max = travel_times.max() if n_tasks > 0 else 0.0
        
        # Spatial features
        if n_tasks > 0:
            locations = task_locations[:n_tasks]
            x_coords = locations[:, 0]
            y_coords = locations[:, 1]
            spatial_span_x = x_coords.max() - x_coords.min() if len(x_coords) > 0 else 0.0
            spatial_span_y = y_coords.max() - y_coords.min() if len(y_coords) > 0 else 0.0
            spatial_center_x = x_coords.mean() if len(x_coords) > 0 else 0.0
            spatial_center_y = y_coords.mean() if len(y_coords) > 0 else 0.0
        else:
            spatial_span_x = spatial_span_y = spatial_center_x = spatial_center_y = 0.0
        
        # Skill coverage features
        skill_coverage = R[:n_tasks, :n_skills].sum(axis=0) if n_tasks > 0 else np.zeros(n_skills)
        robot_skill_diversity = Q[:n_robots, :n_skills].sum(axis=1).mean() if n_robots > 0 else 0.0
        
        # Aggregate features
        aggregate_features = np.array([
            n_tasks, n_robots, n_skills,
            task_exec_mean, task_exec_std, task_exec_max, task_exec_sum,
            travel_mean, travel_max,
            spatial_span_x, spatial_span_y, spatial_center_x, spatial_center_y,
            robot_skill_diversity,
            len(precedence),
            skill_coverage.mean() if len(skill_coverage) > 0 else 0.0,
        ], dtype=np.float32)
        
        return {
            'Q': Q_padded,
            'R': R_padded,
            'T_e': T_e_padded,
            'T_t': T_t_padded,
            'task_locations': task_locations_padded,
            'precedence_matrix': precedence_matrix,
            'task_mask': task_mask,
            'robot_mask': robot_mask,
            'aggregate_features': aggregate_features,
            'n_tasks': n_tasks,
            'n_robots': n_robots,
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data[idx]
        makespan = self.makespans[idx]
        
        return {
            'Q': torch.FloatTensor(features['Q']),
            'R': torch.FloatTensor(features['R']),
            'T_e': torch.FloatTensor(features['T_e']),
            'T_t': torch.FloatTensor(features['T_t']),
            'task_locations': torch.FloatTensor(features['task_locations']),
            'precedence_matrix': torch.FloatTensor(features['precedence_matrix']),
            'task_mask': torch.FloatTensor(features['task_mask']),
            'robot_mask': torch.FloatTensor(features['robot_mask']),
            'aggregate_features': torch.FloatTensor(features['aggregate_features']),
            'makespan': torch.FloatTensor([makespan]),
        }


class MRTANetwork(nn.Module):
    """Neural network for MRTA makespan prediction"""
    
    def __init__(self, max_tasks=10, max_robots=8, max_skills=3, 
                 hidden_dim=128, num_layers=3, dropout=0.2):
        super(MRTANetwork, self).__init__()
        
        self.max_tasks = max_tasks
        self.max_robots = max_robots
        self.max_skills = max_skills
        
        # Encoder for robot skill matrix Q
        self.Q_encoder = nn.Sequential(
            nn.Linear(max_skills, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Encoder for task requirement matrix R
        self.R_encoder = nn.Sequential(
            nn.Linear(max_skills, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Encoder for execution times
        self.T_e_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Encoder for travel time matrix (using 1D conv to capture spatial relationships)
        self.T_t_encoder = nn.Sequential(
            nn.Conv1d(max_tasks, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Encoder for task locations
        self.location_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Encoder for precedence matrix
        self.precedence_encoder = nn.Sequential(
            nn.Conv1d(max_tasks, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Aggregate features encoder
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Pooling layers
        self.task_pool = nn.AdaptiveAvgPool1d(1)
        self.robot_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate total feature dimension
        # Q_flat: max_robots * (hidden_dim // 4)
        # R_flat: max_tasks * (hidden_dim // 4)
        # T_e_flat: max_tasks * (hidden_dim // 4)
        # T_t_flat: max_tasks * (hidden_dim // 4)
        # location_flat: max_tasks * (hidden_dim // 4)
        # precedence_flat: max_tasks * (hidden_dim // 4)
        # aggregate_encoded: hidden_dim // 2
        feature_dim = (hidden_dim // 4) * max_robots + 5 * (hidden_dim // 4) * max_tasks + (hidden_dim // 2)
        
        # Main prediction network
        layers = []
        input_dim = feature_dim
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, x):
        Q = x['Q']  # (batch, max_robots, max_skills)
        R = x['R']  # (batch, max_tasks, max_skills)
        T_e = x['T_e']  # (batch, max_tasks)
        T_t = x['T_t']  # (batch, max_tasks, max_tasks)
        task_locations = x['task_locations']  # (batch, max_tasks, 2)
        precedence_matrix = x['precedence_matrix']  # (batch, max_tasks, max_tasks)
        task_mask = x['task_mask']  # (batch, max_tasks)
        robot_mask = x['robot_mask']  # (batch, max_robots)
        aggregate_features = x['aggregate_features']  # (batch, 16)
        
        batch_size = Q.size(0)
        
        # Encode robot skills
        Q_encoded = self.Q_encoder(Q)  # (batch, max_robots, hidden_dim//4)
        Q_encoded = Q_encoded * robot_mask.unsqueeze(-1)  # Apply mask
        Q_flat = Q_encoded.view(batch_size, -1)  # (batch, max_robots * hidden_dim//4)
        
        # Encode task requirements
        R_encoded = self.R_encoder(R)  # (batch, max_tasks, hidden_dim//4)
        R_encoded = R_encoded * task_mask.unsqueeze(-1)  # Apply mask
        R_flat = R_encoded.view(batch_size, -1)  # (batch, max_tasks * hidden_dim//4)
        
        # Encode execution times
        T_e_expanded = T_e.unsqueeze(-1)  # (batch, max_tasks, 1)
        T_e_encoded = self.T_e_encoder(T_e_expanded)  # (batch, max_tasks, hidden_dim//4)
        T_e_encoded = T_e_encoded * task_mask.unsqueeze(-1)
        T_e_flat = T_e_encoded.view(batch_size, -1)  # (batch, max_tasks * hidden_dim//4)
        
        # Encode travel times
        T_t_encoded = self.T_t_encoder(T_t)  # (batch, hidden_dim//4, max_tasks)
        T_t_encoded = T_t_encoded * task_mask.unsqueeze(1)  # Apply mask
        T_t_flat = T_t_encoded.view(batch_size, -1)  # (batch, max_tasks * hidden_dim//4)
        
        # Encode task locations
        location_encoded = self.location_encoder(task_locations)  # (batch, max_tasks, hidden_dim//4)
        location_encoded = location_encoded * task_mask.unsqueeze(-1)
        location_flat = location_encoded.view(batch_size, -1)  # (batch, max_tasks * hidden_dim//4)
        
        # Encode precedence constraints
        precedence_encoded = self.precedence_encoder(precedence_matrix)  # (batch, hidden_dim//4, max_tasks)
        precedence_encoded = precedence_encoded * task_mask.unsqueeze(1)
        precedence_flat = precedence_encoded.view(batch_size, -1)  # (batch, max_tasks * hidden_dim//4)
        
        # Encode aggregate features
        aggregate_encoded = self.aggregate_encoder(aggregate_features)  # (batch, hidden_dim//2)
        
        # Concatenate all features
        combined = torch.cat([
            Q_flat, R_flat, T_e_flat, T_t_flat, 
            location_flat, precedence_flat, aggregate_encoded
        ], dim=1)
        
        # Predict makespan
        makespan_pred = self.predictor(combined)
        
        return makespan_pred.squeeze(-1)


def load_dataset(data_dir: str, max_samples: int = None) -> Tuple[List[str], List[str]]:
    """Load problem and solution file paths"""
    problem_dir = os.path.join(data_dir, 'problem_instances')
    solution_dir = os.path.join(data_dir, 'solutions')
    
    # Get all problem files
    problem_files = sorted(glob.glob(os.path.join(problem_dir, '*.json')))
    
    # Match with solution files
    problem_files_matched = []
    solution_files_matched = []
    
    for prob_file in problem_files:
        # Extract index from filename
        # Format: problem_instance_1p_000000.json
        basename = os.path.basename(prob_file)
        parts = basename.replace('.json', '').split('_')
        
        if len(parts) >= 3:
            # Try to find matching solution
            # Format: optimal_schedule_1p_000000.json
            idx = parts[-1]
            prefix = parts[-2]  # e.g., '1p'
            
            sol_pattern = os.path.join(solution_dir, f'optimal_schedule_{prefix}_{idx}.json')
            if os.path.exists(sol_pattern):
                problem_files_matched.append(prob_file)
                solution_files_matched.append(sol_pattern)
            else:
                # Try alternative pattern
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


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    """Train the model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            makespan_pred = model(batch)
            loss = criterion(makespan_pred, batch['makespan'].squeeze())
            loss.backward()
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
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_mrta_model.pth')
            print(f'  -> New best model saved! (Val Loss: {val_loss:.4f})')
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            batch = {k: v.to(device) for k, v in batch.items()}
            makespan_pred = model(batch)
            
            predictions.extend(makespan_pred.cpu().numpy())
            targets.extend(batch['makespan'].squeeze().cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    print(f'\nEvaluation Metrics:')
    print(f'  MSE:  {mse:.4f}')
    print(f'  MAE:  {mae:.4f}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  MAPE: {mape:.2f}%')
    
    # Plot predictions vs targets
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('True Makespan')
    plt.ylabel('Predicted Makespan')
    plt.title('Predictions vs True Makespan')
    plt.savefig('predictions_vs_targets.png')
    print('\nSaved prediction plot to predictions_vs_targets.png')
    
    return predictions, targets, {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape}


def plot_training_curves(train_losses, val_losses):
    """Plot training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    print('Saved training curves to training_curves.png')


def main():
    """Main training script"""
    # Configuration
    DATA_DIR = 'dataset_optimal_8t3r3s'
    MAX_SAMPLES = 5000  # Increased for better training (set to None for all)
    BATCH_SIZE = 32
    NUM_EPOCHS = 30  # Increased for better training
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 256  # Increased for better capacity
    NUM_LAYERS = 4  # Increased for better capacity
    DROPOUT = 0.3
    MAX_TASKS = 10
    MAX_ROBOTS = 8
    MAX_SKILLS = 3
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
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
    
    # Create model
    model = MRTANetwork(
        max_tasks=MAX_TASKS,
        max_robots=MAX_ROBOTS,
        max_skills=MAX_SKILLS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    print(f'\nModel architecture:')
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
    model.load_state_dict(torch.load('best_mrta_model.pth'))
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    predictions, targets, metrics = evaluate_model(model, test_loader, device=device)
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()

