# Model Checkpoints Catalog

This directory stores PyTorch model weight checkpoints (`state_dict`) for the trained forecasting architectures across all experimental phases.

All sequence models were trained using PyTorch with:
- Random Seed: `SEED = 42`
- Optimizer: `Adam(lr=0.001, weight_decay=1e-4)`
- Loss Function: `nn.MSELoss()`
- Early Stopping: `patience = 25` epochs on validation partition loss
- Maximum Epochs: `150`
- Target Normalization: `StandardScaler` fitted strictly on the training partition ($N_{\text{train}} = 31$).

---

## Checkpoint Registry

| Checkpoint File | Architecture | Input Dim ($D$) | Input Features | Lookback ($L$) | Horizon ($H$) | Phase / Notebook Reference | Description |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| `gru_best_model.pt` | Compact GRU | 1 | `total_vehicle_count` | 10 | 5 | Phase 10 (`10_gru_forecasting.ipynb`) | **Primary Champion Model**: 1-layer GRU (16 hidden units, dropout=0.10). Lowest overall test MAE (0.7843). |
| `lstm_best_model.pt` | Compact LSTM | 1 | `total_vehicle_count` | 10 | 5 | Phase 9 (`09_lstm_forecasting.ipynb`) | Compact 1-layer LSTM (16 hidden units, dropout=0.10). Test MAE = 1.0543. |
| `gru_lookback_l5.pt` | Compact GRU | 1 | `total_vehicle_count` | 5 | 5 | Phase 16 (`16_lookback_robustness.ipynb`) | Lookback sensitivity condition $L=5$ (22 training windows). Test MAE = 1.0149. |
| `gru_lookback_l20.pt` | Compact GRU | 1 | `total_vehicle_count` | 20 | 5 | Phase 16 (`16_lookback_robustness.ipynb`) | Extended lookback condition $L=20$ (7 training windows due to burn-in). Test MAE = 1.5275. |
| `ablation_config_a.pt` | Compact LSTM | 1 | `total_vehicle_count` | 10 | 5 | Phase 11 (`11_feature_ablation.ipynb`) | Feature Ablation Config A: Univariate traffic count baseline. |
| `ablation_config_b.pt` | Compact LSTM | 2 | `total_vehicle_count`, `heavy_vehicle_ratio` | 10 | 5 | Phase 11 (`11_feature_ablation.ipynb`) | Feature Ablation Config B: Total count + vehicle classification ratio. |
| `ablation_config_c.pt` | Compact LSTM | 2 | `total_vehicle_count`, `spatial_occupancy_ratio` | 10 | 5 | Phase 11 (`11_feature_ablation.ipynb`) | Feature Ablation Config C: Total count + visual roadway occupancy (Lowest MAE among ablation models: 0.9995). |
| `ablation_config_d.pt` | Compact LSTM | 3 | `total_vehicle_count`, `heavy_vehicle_ratio`, `spatial_occupancy_ratio` | 10 | 5 | Phase 11 (`11_feature_ablation.ipynb`) | Feature Ablation Config D: Full multivariate feature suite. |

---

## Loading a Checkpoint in Python

```python
import torch
from src.models import CompactTrafficGRU, load_model_checkpoint

# Initialize architecture matching checkpoint configuration
model = CompactTrafficGRU(
    input_dim=1,
    hidden_dim=16,
    num_layers=1,
    output_dim=5,
    dropout=0.10
)

# Load weights safely
load_model_checkpoint(model, "models/gru_best_model.pt")
model.eval()
```
