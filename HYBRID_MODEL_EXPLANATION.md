# Hybrid GNN + Transformer Model Architecture

## Overview

This model combines the best of both worlds:
- **GNN**: Captures **local** task relationships (precedence, travel times)
- **Transformer**: Models **global** task ordering and long-range dependencies

## Architecture Diagram

```
Input Features
    │
    ├─ Task Requirements (R)
    ├─ Execution Times (T_e)
    ├─ Task Locations (x, y)
    ├─ Travel Times (T_t)
    ├─ Precedence Constraints
    └─ Robot Skills (Q)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              Node Feature Encoder                       │
│  [R, T_e, locations] → hidden_dim embeddings            │
└─────────────────────────────────────────────────────────┘
    │
    ├──────────────────────┬──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   GNN       │    │ Transformer  │    │   Robot      │
│   Branch    │    │   Branch     │    │   Encoder    │
│             │    │              │    │              │
│ Local task  │    │ Global task  │    │ Robot skills │
│ relations   │    │ ordering     │    │              │
│             │    │              │    │              │
│ • Precedence│    │ • Self-      │    │              │
│ • Travel    │    │   attention  │    │              │
│   times     │    │ • Positional │    │              │
│             │    │   encoding   │    │              │
└─────────────┘    └──────────────┘    └──────────────┘
    │                      │                        │
    └──────────┬───────────┘                        │
               │                                    │
               ▼                                    │
        ┌──────────────┐                            │
        │  Fusion      │                            │
        │  Layer       │                            │
        │  (Concat +   │                            │
        │   MLP)       │                            │
        └──────────────┘                            │
               │                                    │
               ▼                                    │
        ┌──────────────┐                            │
        │ Attention-   │                            │
        │ based Pooling│                            │
        │ (Graph-level)│                            │
        └──────────────┘                            │
               │                                    │
               └──────────┬─────────────────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │  Concatenate │
                  │  All Features│
                  └──────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │   Predictor  │
                  │   (MLP)      │
                  └──────────────┘
                          │
                          ▼
                    Makespan
```

## Key Components

### 1. **GNN Branch (Local Relationships)**

**Purpose**: Capture immediate task dependencies

**How it works**:
- Tasks are **nodes** in a graph
- **Edges** created from:
  - Precedence constraints (directed)
  - Travel times (weighted, symmetric)
- Graph convolution propagates information along edges
- Learns which tasks are directly related

**Example**:
```
Task 1 →[precedence]→ Task 2 →[travel]→ Task 3
```
GNN learns: "Task 1 must come before Task 2, and Task 2 is close to Task 3"

---

### 2. **Transformer Branch (Global Ordering)**

**Purpose**: Model long-range dependencies and task sequences

**How it works**:
- Self-attention mechanism
- Each task attends to **all other tasks**
- Positional encoding adds ordering information
- Learns global task patterns

**Example**:
```
Task 1 can attend to Task 5, Task 8, etc. (even if not directly connected)
```
Transformer learns: "Tasks 1, 5, 8 form a critical path"

---

### 3. **Fusion Layer**

**Purpose**: Combine local (GNN) and global (Transformer) information

**How it works**:
- Concatenates GNN output + Transformer output
- MLP learns how to combine them
- Each task gets enriched representation

**Result**: 
- Task embeddings contain both:
  - Local neighborhood info (from GNN)
  - Global context (from Transformer)

---

### 4. **Attention-Based Pooling**

**Purpose**: Aggregate task-level features to graph-level

**How it works**:
- Learns which tasks are most important
- Weighted sum based on attention scores
- More important tasks contribute more to final prediction

---

## Why This Works Better

### Problem with GNN alone:
- ❌ Limited receptive field (only immediate neighbors)
- ❌ May miss long-range dependencies

### Problem with Transformer alone:
- ❌ Doesn't explicitly model graph structure
- ❌ May not capture local precedence well

### Hybrid Solution:
- ✅ **GNN** handles local structure (precedence, travel)
- ✅ **Transformer** handles global patterns (critical paths)
- ✅ **Fusion** combines both for best representation
- ✅ **Attention pooling** focuses on important tasks

---

## Expected Improvements

Compared to baseline model:

1. **Better MAPE**: 10-12% (vs 16.73%)
2. **Reduced bias**: Better handling of high makespan
3. **Better generalization**: Captures both local and global patterns
4. **More interpretable**: Attention weights show which tasks matter

---

## Hyperparameters

```python
HIDDEN_DIM = 256          # Feature dimension
NUM_GNN_LAYERS = 2        # Local relationship depth
NUM_TRANSFORMER_LAYERS = 2 # Global attention depth
NUM_HEADS = 8             # Multi-head attention
D_FF = 512                # Feed-forward dimension
DROPOUT = 0.3            # Regularization
```

---

## Training Tips

1. **Start with fewer layers**: 2 GNN + 2 Transformer is good
2. **Monitor attention weights**: Can visualize which tasks are important
3. **Use learning rate scheduling**: Transformer benefits from it
4. **More data helps**: Transformer needs more samples than GNN alone

---

## Comparison with Other Models

| Model | Local Relations | Global Patterns | Complexity | Data Needed |
|-------|----------------|-----------------|------------|-------------|
| **Baseline MLP** | ❌ | ❌ | Low | Low |
| **GNN** | ✅ | ❌ | Medium | Medium |
| **Transformer** | ❌ | ✅ | High | High |
| **Hybrid (GNN+Transformer)** | ✅ | ✅ | High | High |

**Best for**: Complex problems with both local constraints and global optimization needs.

