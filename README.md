# Vision-Based Traffic Flow Prediction using YOLOv8, ByteTrack, and Recurrent Neural Networks

## 1. Overview

Accurate urban traffic prediction is a cornerstone of intelligent transportation systems (ITS), enabling adaptive signal control, route guidance, and congestion mitigation. Traditional traffic monitoring relies heavily on embedded physical sensors, such as inductive loops and radar detectors, which are expensive to install and maintain and provide limited spatial context. By contrast, existing closed-circuit television (CCTV) surveillance infrastructure provides rich visual data that can be leveraged for non-invasive, continuous traffic perception.

This project implements an end-to-end, reproducible machine learning pipeline that transforms raw surveillance video into short-horizon multi-step traffic volume forecasts. The system bridges perception and sequence modeling in two distinct phases: first, extracting frame-level traffic states using **YOLOv8** object detection, **ByteTrack** multi-object tracking, and roadway **Region of Interest (ROI) spatial occupancy** estimation; and second, modeling temporal traffic dynamics across sequence windows using **Compact LSTM** and **Compact GRU** recurrent neural networks alongside statistical baselines.

The primary objective of this project is to build an empirical, leakage-safe pipeline that evaluates whether vision-derived features enhance short-term sequence forecasting under rigorous chronological partitioning, while subjecting the models to multi-horizon error analysis, regime-transition stress tests, and conformal uncertainty quantification.

---

## 2. Project Pipeline

The pipeline follows a modular sequence from raw video frames to evaluated multi-step predictions:

```mermaid
flowchart LR
    A[Surveillance Video] --> B[YOLOv8 Detection]
    B --> C[ByteTrack Multi-Object Tracking]
    C --> D[ROI & Spatial Occupancy Extraction]
    D --> E[Chronological Time Series Construction]
    E --> F[Leakage-Safe Partitioning]
    F --> G[Sequence Models: LSTM / GRU / Baselines]
    G --> H[Evaluation & Uncertainty Calibration]
```

1. **Traffic Video**: Ingestion of raw surveillance clips from a fixed highway perspective.
2. **Detection**: Vehicle localization and classification using YOLOv8.
3. **Tracking**: Tracklet association and trajectory persistence using ByteTrack.
4. **Traffic-State Extraction**: Directional line-crossing counting and spatial roadway occupancy.
5. **Time-Series Construction**: Aggregation of clip-level observations into structured time series.
6. **Sequence Modeling**: Multi-step forecasting using compact recurrent architectures and baseline forecasters.
7. **Evaluation & Uncertainty**: Multi-horizon diagnostics, regime analysis, and conformal prediction intervals.

---

## 3. Dataset

The project utilizes surveillance video from the **UCSD Traffic Dataset** (freeway surveillance along Interstate 5 in San Diego, captured at $320 \times 240$ resolution at 10 frames per second):

- **Data Structure**: The repository uses $N = 44$ surveillance video clips sampled across varying daylight conditions and traffic densities.
- **Discrete Sampling Characteristic**: Rather than a single continuous recording, these video clips represent **discrete sampled temporal observations**. Each clip captures approximately 45–52 seconds of traffic, from which representative traffic state features (mean vehicle count, heavy vehicle ratio, spatial occupancy) are aggregated.
- **Traffic Regimes**: The dataset spans three operational conditions: light (free-flow), medium (moderate flow), and heavy (congested / stop-and-go) traffic states.

---

## 4. Methodology

### 4.1. YOLOv8 Vehicle Detection
Vehicle localization employs a pretrained YOLOv8 Nano (`yolov8n.pt`) detector. Detections are filtered for relevant COCO vehicle classes: **car** (class 2), **bus** (class 5), and **truck** (class 7), with an empirical confidence threshold of $\tau = 0.25$.

### 4.2. ByteTrack Multi-Object Tracking
To associate bounding boxes across successive video frames, the pipeline integrates **ByteTrack**, which preserves track identity across low- and high-confidence detections via Kalman filtering and Hungarian matching. 
- *Per-Clip State Isolation*: Tracking state, active tracklet IDs, and trajectory buffers are strictly re-initialized per video clip to prevent identity bleeding across non-contiguous surveillance clips.

### 4.3. ROI-Based Vehicle Counting & Virtual Gate
Traffic volume is measured through two complementary visual rules:
1. **Centroid-in-ROI Rule**: Vehicles are counted if their bounding-box centroid falls within a calibrated 5-point roadway polygon bounding active travel lanes.
2. **Virtual Line-Crossing Gate**: Southbound flow is captured as vehicle centroids cross a virtual horizontal reference line ($y = 160$ px, $x \in [100, 310]$).

### 4.4. Roadway Spatial Occupancy Extraction
To capture visual traffic congestion independent of discrete vehicle counts, roadway spatial occupancy is computed as the pixel-wise intersection between vehicle masks and the roadway ROI:
$$\text{Occupancy Ratio} = \frac{\text{Area}\left(\bigcup_{i} B_i \cap \mathcal{M}_{\text{ROI}}\right)}{\text{Area}(\mathcal{M}_{\text{ROI}})}$$
where $B_i$ denotes the bounding box of detection $i$, and $\mathcal{M}_{\text{ROI}}$ is the binary roadway mask ($28{,}562$ pixels).

### 4.5. Time-Series Formulation
Per-clip summaries are compiled into a 44-step chronological sequence ($t = 1, \dots, 44$). Features include:
- `total_vehicle_count` (primary target series)
- `heavy_vehicle_ratio` (proportion of buses and trucks)
- `spatial_occupancy_ratio` (fraction of roadway area occupied by vehicles)

### 4.6. Recurrent Sequence Architectures (LSTM & GRU)
To mitigate overfitting on short sequence lengths ($N=44$), compact single-layer recurrent architectures were designed:
- **CompactTrafficLSTM**: 1-layer LSTM ($16$ hidden units, dropout $p = 0.10$, $1{,}285$ parameters) projecting the final hidden state via a linear head to multi-step horizon $H = 5$.
- **CompactTrafficGRU**: 1-layer GRU ($16$ hidden units, dropout $p = 0.10$, $1{,}029$ parameters) projecting to $H = 5$.
- **Statistical Baselines**: Naïve Persistence ($y_{t+h} = y_t$), Historical Mean ($\bar{y}_{\text{train}}$), and Simple Moving Averages ($\text{SMA-}3$, $\text{SMA-}5$).

### 4.7. Leakage-Safe Chronological Splitting
To guarantee temporal integrity:
- **Partitioning**: Strictly chronological split into **Train** ($N_{\text{train}} = 31$, $70.5\%$), **Validation** ($N_{\text{val}} = 7$, $15.9\%$), and **Test** ($N_{\text{test}} = 6$, $13.6\%$) sets with zero shuffling.
- **Normalization**: `StandardScaler` instances are fitted exclusively on the training partition; validation and test partitions are transformed using training statistics.
- **Target-Anchored Evaluation Windows**: Sliding windows are defined such that all forecast targets fall entirely within partition boundaries. For lookback $L = 10$ and horizon $H = 5$, this yields exactly $M_{\text{val}} = 3$ validation windows and $M_{\text{test}} = 2$ test windows (10 horizon observation points) shared identically across all models.

---

## 5. Experiments & Evaluation

All evaluations are conducted under identical multi-step target horizons ($H = 5$, representing forecast steps $H_1$ through $H_5$). Evaluation metrics include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Empirical Conformal Coverage, and Average Interval Width (AIW).

### 5.1. Model Benchmark Comparison (Out-of-Sample Test Set)

Evaluated across strictly identical test target windows ($M_{\text{test}} = 2$ windows, 10 horizon evaluation steps):

| Model | Model Type | Parameters | Overall MAE | Overall RMSE | $H_1$ MAE | $H_5$ MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Mean** | Baseline | 0 | 6.9241 | 6.9722 | 7.4077 | 6.6742 |
| **Persistence (Naïve)** | Baseline | 0 | 0.8603 | 1.0612 | 0.9215 | 0.4580 |
| **SMA-3** | Baseline | 0 | 0.6732 | 0.8814 | 0.2712 | 1.1083 |
| **SMA-5** | Baseline | 0 | 0.6860 | 0.9070 | 0.2599 | 1.1196 |
| **Compact LSTM** | Deep RNN | 1,285 | 1.0181 | 1.2587 | 0.8994 | 1.1813 |
| **Compact GRU** | Deep RNN | **1,029** | **0.7843** | **1.1713** | **0.2706** | **1.6977** |

*Findings*:
- **Champion Deep Model**: Compact GRU achieved the lowest MAE among deep learning models (0.7843), achieving high short-range precision on $H_1$ (MAE: 0.2706).
- **Strong Statistical Baselines**: Simple Moving Averages (SMA-3 MAE: 0.6732) proved very competitive on this small sample, serving as a vital reality check against over-parameterization.

### 5.2. Feature Ablation Study

Evaluating whether auxiliary visual perception features improve LSTM forecasting performance:

| Configuration | Input Features | Feature Dim ($D$) | Val MAE | Test MAE | Test RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Config A** | Total Count | 1 | **0.5630** | 1.0181 | 1.2587 |
| **Config B** | Count + Heavy Vehicle Ratio | 2 | 0.6809 | **0.9488** | 1.3481 |
| **Config C** | Count + Spatial Occupancy | 2 | 0.7028 | 1.1565 | 1.3116 |
| **Config D** | Count + HVR + Spatial Occupancy | 3 | 0.6221 | 1.0203 | 1.3762 |

*Findings*:
- On the validation set, univariate Count (Config A) achieved the lowest error.
- On the out-of-sample test set, adding Heavy Vehicle Ratio (Config B) reduced test MAE to 0.9488, while adding full features (Config D) caused slight over-fitting due to the limited sequence length.

### 5.3. Multi-Horizon Error Drift & Degradation

Forecast error increases across horizon steps as uncertainty accumulates recursively:
- **Compact GRU**: $H_1$ MAE = 0.2706 $\to$ $H_2$ MAE = 0.3330 $\to$ $H_3$ MAE = 0.5260 $\to$ $H_4$ MAE = 1.0944 $\to$ $H_5$ MAE = 1.6977 ($+527.4\%$ drift).
- **Persistence**: Stable across steps because traffic volumes in this test period exhibited a moderate rebound towards the mean.

### 5.4. Transition-Aware Evaluation (Stable vs. Rapid Transitions)

Test horizons were categorized based on prior transition magnitude $\Delta y = |y_t - y_{t-1}|$:
- **Stable Flow**: Compact GRU demonstrated strong tracking performance during stable conditions, achieving an overall MAE of 0.5939 ($H_1$ MAE = 0.0081).
- **Rapid Transitions**: During sharp volume swings, GRU error climbed to MAE = 0.9747 ($H_5$ MAE = 2.8221), indicating that purely autoregressive sequence models lag behind abrupt step-level transitions.

### 5.5. Conformal Uncertainty Quantification

Using inductive conformal prediction, absolute validation residuals were used to calibrate horizon-specific nonconformity quantiles $q_h = \max_{i \in \text{Val}} |e_{i, h}|$:
- **Empirical Test Coverage**: 100% on $H_1$, 100% on $H_2$, 0% on $H_3$, 50% on $H_4$, and 50% on $H_5$, yielding an aggregate coverage of 60.0%.
- **Physical Bounds**: Prediction intervals $[ \hat{y} - q_h, \hat{y} + q_h ]$ were clipped at zero, providing realistic non-negative traffic ranges.

### 5.6. Lookback Sensitivity & Robustness ($L \in \{5, 10, 20\}$)

| Lookback ($L$) | Train Windows | Val Windows | Test Windows | Val MAE | Test MAE | Test RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$L = 5$** | 22 | 3 | 2 | 0.7941 | 1.0149 | 1.4337 |
| **$L = 10$** | 17 | 3 | 2 | **0.5899** | **0.7843** | **1.1713** |
| **$L = 20$** | 7 | 3 | 2 | 1.5461 | 1.5275 | 1.6601 |

*Methodological Finding*: $L = 10$ provided the optimal balance. Crucially, the degradation at $L = 20$ cannot be attributed solely to longer context length: because of burn-in requirements on a finite dataset of $N=44$, increasing $L$ from 10 to 20 reduced available training sequences from 17 to 7, introducing sample starvation as a primary confounder.

---

## 6. Repository Structure

```text
traffic-flow-yolo-lstm/
├── configs/
│   └── config.yaml               # Centralized pipeline configuration & parameters
├── src/                          # Reusable core modules
│   ├── __init__.py               # Package metadata, seed setting, config loader
│   ├── detection.py              # YOLOv8 vehicle detection & bounding box filtering
│   ├── tracking.py               # ByteTrack tracker wrapper & line-crossing counter
│   ├── preprocessing.py          # ROI masking, spatial occupancy, scalers, temporal split
│   ├── datasets.py               # Target-anchored sequence windowing & PyTorch DataLoaders
│   ├── models.py                 # Compact LSTM/GRU architectures & baseline models
│   └── evaluation.py             # Multi-horizon metrics & conformal calibration
├── tests/
│   └── test_smoke.py             # Automated pytest suite validating imports, models, & outputs
├── notebooks/                    # 16-step experimental narrative (EDA to robustness)
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_yolo_detection.ipynb
│   ├── ...
│   └── 16_lookback_robustness.ipynb
├── models/                       # Trained PyTorch checkpoint catalog (.pt)
│   ├── gru_best_model.pt         # Champion GRU model weights
│   ├── lstm_best_model.pt        # Compact LSTM model weights
│   ├── README.md                 # Architecture parameters and checkpoint registry
│   └── ...
├── outputs/                      # Generated experimental artifacts
│   ├── figures/                  # 30+ publication-ready diagnostic plots (.png)
│   ├── tables/                   # Benchmark CSVs for metrics, ablation, and robustness
│   └── predictions/              # Saved test forecast predictions and prediction intervals
├── requirements.txt              # Production Python dependencies
├── .gitignore                    # Git tracking rules
└── README.md                     # Project documentation
```

### Directory Overview:
- `src/`: Production-grade library modules implementing each pipeline stage.
- `notebooks/`: Numbered, fully executable experimental walkthroughs from video exploration to sensitivity analysis.
- `configs/`: Single source of truth (`config.yaml`) defining random seeds, hyperparameters, ROI coordinates, and paths.
- `models/`: Saved PyTorch model weights (`state_dict`) for trained LSTM and GRU configurations.
- `outputs/`: Complete set of precomputed benchmark metrics, predictions, and diagnostic figures.
- `tests/`: Automated unit and smoke tests executed via `pytest`.

---

## 7. Reproducibility

### 7.1. Environment Setup

Clone the repository and install dependencies in an isolated virtual environment:

```bash
# 1. Clone repository
git clone https://github.com/username/traffic-flow-yolo-lstm.git
cd traffic-flow-yolo-lstm

# 2. Create and activate virtual environment (Python 3.9 - 3.12 recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 7.2. Automated Testing

Verify the codebase, configurations, model forward passes, and output files:

```bash
pytest tests/test_smoke.py -v
```

### 7.3. Centralized Configuration

All pipeline parameters (random seed 42, model dimensions, learning rate, ROI polygon vertices, split ratios) are managed through `configs/config.yaml`. To inspect or modify configurations programmatically:

```python
from src import load_config, set_seed

cfg = load_config()
set_seed(cfg["project"]["seed"])
```

### 7.4. Running Pretrained Models

To load and evaluate saved checkpoints:

```python
import torch
from src.models import CompactTrafficGRU, load_model_checkpoint

model = CompactTrafficGRU(input_dim=1, hidden_dim=16, output_dim=5)
model = load_model_checkpoint(model, "models/gru_best_model.pt")
model.eval()

# Forward pass on dummy batch (batch_size=1, lookback=10, features=1)
sample_input = torch.randn(1, 10, 1)
forecast = model(sample_input)
print("5-step forecast output shape:", forecast.shape)  # torch.Size([1, 5])
```

---

## 8. Limitations & Future Work

### Genuine Limitations:
1. **Sample Size Constraints**: With $N = 44$ discrete observations, the sequence dataset is inherently compact, requiring strong capacity constraints ($16$ hidden units, single layer, dropout) to prevent overfitting.
2. **Discrete Sampling Frequency**: Surveillance clips represent intermittent time-slice samples rather than a high-frequency continuous feed, meaning inter-clip dynamics can exhibit non-Markovian jumps that challenge purely autoregressive models.
3. **Monocular Viewpoint**: Spatial occupancy is calculated in image pixel space without inverse perspective mapping (IPM) or homography transformation to bird's-eye-view road coordinates.

### Future Improvements:
1. **Continuous Video Streaming**: Evaluating the pipeline on 24-hour continuous surveillance data to capture rich diurnal seasonality and peak-hour rhythms.
2. **Bird's-Eye-View (BEV) Homography**: Projecting vehicle bounding boxes onto calibrated physical roadway planes to measure physical vehicle densities ($\text{veh/km/lane}$).
3. **Probabilistic & Spatio-Temporal Models**: Extending sequence architectures to probabilistic forecasting (e.g., DeepAR, Conformal Quantile Regression) or Spatio-Temporal Graph Neural Networks (ST-GNNs) across multi-camera networks.

---

## 9. Conclusion

This project demonstrates a rigorous, end-to-end integration of **computer vision perception** and **sequence-to-sequence temporal forecasting**:
- **Computer Vision Perception**: Real-time YOLOv8 vehicle detection and ByteTrack multi-object tracking.
- **Physical Traffic Metrics**: Roadway ROI spatial occupancy extraction and directional line counting.
- **Leakage-Safe Machine Learning**: Strict chronological train/validation/test partitioning and target-anchored evaluation windows.
- **Deep Sequence Modeling**: Compact LSTM and GRU networks benchmarked rigorously against established statistical baselines.
- **Diagnostic Rigor**: Multi-horizon degradation analysis, regime-transition stress testing, lookback sensitivity analysis, and conformal uncertainty quantification.
- **Reproducible ML Engineering**: Centralized YAML configuration, modular Python architecture, automated pytest test suite, and cataloged model weights.
