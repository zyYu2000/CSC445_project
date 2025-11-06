"""
Performance Visualization Utilities
Can be imported by any model for comprehensive performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None


def plot_comprehensive_performance(predictions, targets, metrics, model_name="Model", save_path="performance_analysis.png"):
    """Create comprehensive performance visualization with 9 subplots"""
    
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
    if SCIPY_AVAILABLE and stats is not None:
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
    bin_errors = [abs_errors[bin_indices == i].mean() if np.any(bin_indices == i) else 0 
                  for i in range(1, len(bins))]
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

