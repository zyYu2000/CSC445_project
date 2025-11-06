# Performance Analysis & Improvements

## Current Performance Issues

### 1. **Low R² Score (0.19)**
- Model explains only **19% of variance**
- Indicates model is missing key patterns
- Suggests insufficient feature learning

### 2. **Strong Prediction Bias**
- **Regression slope: 0.135** (should be ~1.0)
- **Underpredicts high makespan** (600-1000 range)
- **Overpredicts low makespan** (300-500 range)
- Clear non-linear bias pattern

### 3. **High Error at Extremes**
- MAE ~240 for makespan 900-1000
- MAE ~160 for makespan 400-500
- Model struggles with edge cases

### 4. **Limited Data**
- Only 500 samples (320 train, 80 val, 100 test)
- Complex model (1.6M parameters) needs more data
- High risk of overfitting

## Root Causes

### 1. **Loss Function**
- Using **MSE Loss** which:
  - Doesn't penalize bias explicitly
  - Sensitive to outliers
  - Doesn't address over/under prediction asymmetry

### 2. **Data Size**
- 500 samples is too small for 1.6M parameter model
- Rule of thumb: Need 10-100 samples per parameter
- Current: ~0.3 samples per parameter ❌

### 3. **Feature Engineering**
- May not be capturing all relevant patterns
- Aggregate features might be insufficient
- Graph structure might not be fully utilized

### 4. **Model Complexity**
- Model might be too complex for available data
- Risk of learning noise instead of signal

## Improvements Implemented

### 1. **Better Loss Function: Huber-Quantile Loss**
```python
# Combines:
- Quantile Loss: Reduces bias (penalizes over/under differently)
- Huber Loss: Robust to outliers
```

**Benefits:**
- ✅ Explicitly addresses prediction bias
- ✅ Less sensitive to outliers
- ✅ Better for non-normal error distributions

### 2. **More Training Data**
- Increased from **500 → 2000 samples**
- Better train/val/test split
- More samples per parameter

### 3. **Improved Hyperparameters**
- Lower learning rate (0.0005 vs 0.001)
- Less dropout (0.25 vs 0.3)
- More epochs (40 vs 30)
- Better weight decay

### 4. **Better Training Strategy**
- Monitor both loss and MSE
- Save best model based on validation MSE
- Improved learning rate scheduling

## Expected Improvements

### With These Changes:

1. **R² Score**: Should improve to **0.4-0.6** (from 0.19)
2. **MAPE**: Should drop to **12-14%** (from 16.14%)
3. **Bias**: Regression slope should be **0.7-0.9** (from 0.135)
4. **Error Distribution**: More uniform across makespan ranges

## Additional Recommendations

### If Still Not Satisfactory:

1. **Use Even More Data**
   - Try 5000-10000 samples
   - Model complexity justifies it

2. **Feature Engineering**
   - Add more aggregate features
   - Include interaction terms
   - Normalize input features

3. **Ensemble Methods**
   - Train multiple models
   - Average predictions
   - Reduces variance

4. **Simpler Model**
   - Reduce hidden dimensions
   - Fewer layers
   - Better for limited data

5. **Different Architecture**
   - Try pure GNN (no transformer)
   - Or pure Transformer (no GNN)
   - See which works better

6. **Transfer Learning**
   - Pre-train on larger dataset
   - Fine-tune on your data

## How to Run Improved Model

```bash
python mrta_improved_model.py
```

This will:
- Use 2000 samples (vs 500)
- Train with Huber-Quantile loss
- Generate improved performance plots
- Save to `improved_model_performance.png`

## Comparison

| Metric | Original | Improved (Expected) |
|--------|----------|---------------------|
| R² Score | 0.19 | 0.4-0.6 |
| MAPE | 16.14% | 12-14% |
| Regression Slope | 0.135 | 0.7-0.9 |
| MAE (high makespan) | ~240 | ~150 |
| Correlation | 0.49 | 0.7-0.8 |

## Next Steps

1. **Run improved model** and compare results
2. **If still poor**: Increase to 5000+ samples
3. **If still poor**: Simplify model architecture
4. **If still poor**: Try different architectures (pure GNN, pure Transformer)
5. **Consider**: This might be a fundamentally hard problem - makespan prediction from problem structure alone is challenging

