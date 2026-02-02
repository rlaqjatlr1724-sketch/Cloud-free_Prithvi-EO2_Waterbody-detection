<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.4-76B900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Research-orange.svg" alt="Status">
</p>

<h1 align="center">☁️ Cloud-Free Water Body Detection</h1>
<h3 align="center">Fine-Tuning NASA's Prithvi-EO-2.0 Foundation Model for<br>Cloud Removal and Multi-Modal SAR-Optical Fusion</h3>

<p align="center">
  <b>Beomsik Kim</b> and <b>Prof. Hyunglok Kim</b><br>
  <i>GIST Hydro AI Lab</i><br>
  School of Environment and Energy Engineering<br>
  Gwangju Institute of Science and Technology, Republic of Korea
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Model Components](#-model-components)
- [Usage](#-usage)
- [Experiments](#-experiments)
- [Results Visualization](#-results-visualization)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 🌍 Overview

### Problem Statement

Cloud contamination significantly impairs the usability of optical satellite imagery, affecting critical applications such as:
- **Environmental monitoring** 🌱
- **Disaster response** 🚨
- **Land-use analysis** 🏙️
- **Hydrological forecasting** 💧

Traditional spectral indices like **MNDWI** (Modified Normalized Difference Water Index) suffer catastrophic performance degradation under cloudy conditions, making reliable water body monitoring nearly impossible during monsoon seasons or in tropical regions.

### Our Solution

We present a **two-stage deep learning pipeline** that achieves **cloud-robust water body detection** by:

1. **Cloud Removal Stage**: Fine-tuning NASA's Prithvi-EO-2.0 foundation model with a Siamese U-Net encoder to reconstruct cloud-obscured optical satellite imagery
2. **Water Detection Stage**: Multi-modal late fusion of restored optical (6 bands) and SAR imagery with learnable attention-based modality weighting

### Key Contributions

- ✅ **26.7% improvement** in Water IoU after cloud removal (0.435 → 0.551)
- ✅ **98% false positive reduction** (1,559 → 29 pixels)
- ✅ **Robust performance** under extreme cloud coverage (0-100%)
- ✅ **Adaptive modality fusion** that automatically relies more on SAR in cloud-degraded regions

---

## 📊 Key Results

### Cloud Coverage vs. Water Detection Performance

<p align="center">
  <img src="cloud_vs_iou_comparison.png" alt="Cloud Coverage vs IoU" width="800">
</p>

| Metric | Our Model | MNDWI (Baseline) | Improvement |
|--------|-----------|------------------|-------------|
| **Water IoU @ 0% cloud** | 0.946 | 0.933 | +1.4% |
| **Water IoU @ 50% cloud** | 0.941 | 0.520 | +81.0% |
| **Water IoU @ 100% cloud** | 0.889 | 0.000 | ∞ |
| **Mean IoU (0-100%)** | 0.935 | 0.498 | +87.8% |

> 💡 **Key Insight**: While MNDWI degrades linearly with cloud coverage, our fusion-based model maintains **>0.9 mIoU** even at 90% cloud coverage.

### Cloud Removal Impact

| Condition | Before Restoration | After Restoration | Change |
|-----------|-------------------|-------------------|--------|
| Accuracy | 0.915 | 0.946 | +3.4% |
| F1-Score | 0.606 | 0.711 | +17.3% |
| **Water IoU** | **0.435** | **0.551** | **+26.7%** |
| False Positives | 1,559 | 29 | -98.1% |

---

## 🏗️ Architecture

### Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLOUD-ROBUST WATER DETECTION PIPELINE                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT                                                                     │
│   ─────                                                                     │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│   │  Cloudy    │  │  Cloudy    │  │  Cloudy    │  │  Cloudy    │           │
│   │  Frame 1   │  │  Frame 2   │  │  Frame 3   │  │  Frame 4   │           │
│   │ (Sentinel-2)│  │ (Sentinel-2)│  │ (Sentinel-2)│  │ (Sentinel-2)│       │
│   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘           │
│         │              │              │              │                     │
│         └──────────────┼──────────────┼──────────────┘                     │
│                        │              │                                     │
│                        ▼              ▼                                     │
│   ┌──────────────────────────────────────────────────────────┐             │
│   │              STAGE 1: CLOUD REMOVAL                       │             │
│   │                                                          │             │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │             │
│   │  │   Siamese   │───▶│   Prithvi   │───▶│   U-Net     │   │             │
│   │  │ UNet Encoder│    │  Encoder    │    │  Decoder    │   │             │
│   │  │(Weight-Shared)│   │(Temporal Attn)│   │ (4 Outputs) │   │             │
│   │  └─────────────┘    └─────────────┘    └─────────────┘   │             │
│   │         ▲                                    │           │             │
│   │         │                                    ▼           │             │
│   │    Skip Connections              ┌─────────────────┐     │             │
│   │    (Multi-Scale)                 │ Restored Frames │     │             │
│   │                                  │  (Cloud-Free)   │     │             │
│   │                                  └────────┬────────┘     │             │
│   └──────────────────────────────────────────│───────────────┘             │
│                                              │                              │
│                                              ▼                              │
│   ┌──────────────────────────────────────────────────────────┐             │
│   │              STAGE 2: WATER DETECTION                     │             │
│   │                                                          │             │
│   │  ┌───────────────┐                ┌───────────────┐      │             │
│   │  │ Optical UNet  │                │   SAR UNet    │      │             │
│   │  │   Encoder     │                │   Encoder     │      │             │
│   │  │  (6 channels) │                │  (2 channels) │      │             │
│   │  └───────┬───────┘                └───────┬───────┘      │             │
│   │          │                                │              │             │
│   │          │      ┌───────────────┐        │              │             │
│   │          └─────▶│   Attention   │◀───────┘              │             │
│   │                 │    Fusion     │                       │             │
│   │                 │   (Learned    │◀── Cloud Probability  │             │
│   │                 │   Weighting)  │       (Hint)          │             │
│   │                 └───────┬───────┘                       │             │
│   │                         │                               │             │
│   │                         ▼                               │             │
│   │                 ┌───────────────┐                       │             │
│   │                 │  UNet Decoder │                       │             │
│   │                 └───────┬───────┘                       │             │
│   │                         │                               │             │
│   └─────────────────────────│───────────────────────────────┘             │
│                             ▼                                              │
│   OUTPUT                                                                   │
│   ──────                                                                   │
│   ┌────────────────────────────────────────────────────────┐              │
│   │              WATER BODY SEGMENTATION MASK               │              │
│   │                   (Binary: Water/Non-Water)             │              │
│   └────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Cloud Removal (Siamese U-Prithvi)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SIAMESE U-PRITHVI ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    T1 ──┐                                                                   │
│    T2 ──┼──▶ Siamese Encoder ──┬──▶ Skip Reduce ──▶ E1_agg (64ch)          │
│    T3 ──┤    (Shared Weights)  ├──▶ Skip Reduce ──▶ E2_agg (128ch)         │
│    T4 ──┘                      ├──▶ Skip Reduce ──▶ E3_agg (256ch)         │
│                                └──▶ Skip Reduce ──▶ E4_agg (512ch)         │
│                                              │                              │
│                                              ▼                              │
│                               ┌──────────────────────────┐                  │
│                               │  Prithvi-EO-2.0 Encoder  │                  │
│                               │  ├─ Patch Embedding      │                  │
│                               │  ├─ Temporal Attention   │                  │
│                               │  └─ Spatial Attention    │                  │
│                               │      (1024-dim latent)   │                  │
│                               └────────────┬─────────────┘                  │
│                                            │                                │
│                                            ▼                                │
│                               ┌──────────────────────────┐                  │
│                               │    Prithvi Decoder       │                  │
│                               │  + UNet Decoder Fusion   │                  │
│                               │  (Skip Connections)      │                  │
│                               └────────────┬─────────────┘                  │
│                                            │                                │
│                                            ▼                                │
│                                ┌─────────────────────────┐                  │
│                                │  4 Restored Frames      │                  │
│                                │  (6 channels each)      │                  │
│                                └─────────────────────────┘                  │
│                                                                             │
│    Loss: L1 + SSIM (Perceptual)                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 2: Multi-Modal Water Detection (Attention Fusion)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION FUSION WATER DETECTION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐                     ┌─────────────────┐               │
│   │   Sentinel-2    │                     │   Sentinel-1    │               │
│   │   (Restored)    │                     │     (SAR)       │               │
│   │   6 channels:   │                     │   2 channels:   │               │
│   │   B,G,R,NIR,    │                     │     VV, VH      │               │
│   │   SWIR1,SWIR2   │                     │                 │               │
│   └────────┬────────┘                     └────────┬────────┘               │
│            │                                       │                        │
│            ▼                                       ▼                        │
│   ┌─────────────────┐                     ┌─────────────────┐               │
│   │  Optical UNet   │                     │    SAR UNet     │               │
│   │    Encoder      │                     │    Encoder      │               │
│   │                 │                     │                 │               │
│   │  E1: 64ch       │                     │  E1: 64ch       │               │
│   │  E2: 128ch      │                     │  E2: 128ch      │               │
│   │  E3: 256ch      │                     │  E3: 256ch      │               │
│   │  E4: 512ch      │                     │  E4: 512ch      │               │
│   │  BN: 1024ch     │                     │  BN: 1024ch     │               │
│   └────────┬────────┘                     └────────┬────────┘               │
│            │                                       │                        │
│            └───────────┬───────────────────────────┘                        │
│                        │                                                    │
│                        ▼                                                    │
│            ┌───────────────────────────┐                                    │
│            │    ATTENTION FUSION       │                                    │
│            │                           │                                    │
│            │  Input: Opt + SAR + Cloud │                                    │
│            │         (C + C + 1)ch     │                                    │
│            │            ↓              │                                    │
│            │    Conv → BN → ReLU       │                                    │
│            │         (C)ch            │                                    │
│            │            ↓              │                                    │
│            │    Conv → BN → ReLU       │                                    │
│            │       (C/2)ch            │                                    │
│            │            ↓              │                                    │
│            │      Conv (1×1)           │                                    │
│            │         (2)ch            │                                    │
│            │            ↓              │                                    │
│            │        Softmax            │                                    │
│            │     [w_opt, w_sar]        │                                    │
│            │            ↓              │                                    │
│            │  Fused = w_opt * Opt      │                                    │
│            │        + w_sar * SAR      │                                    │
│            │                           │                                    │
│            └───────────┬───────────────┘                                    │
│                        │                                                    │
│                        ▼                                                    │
│            ┌───────────────────────────┐                                    │
│            │      UNet Decoder         │                                    │
│            │  (Fused Skip Connections) │                                    │
│            └───────────┬───────────────┘                                    │
│                        │                                                    │
│                        ▼                                                    │
│            ┌───────────────────────────┐                                    │
│            │    Water Probability      │                                    │
│            │      (Sigmoid Output)     │                                    │
│            └───────────────────────────┘                                    │
│                                                                             │
│   Loss: BCE (0.5) + Lovász-Softmax (0.5)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA 12.4+ (for GPU acceleration)
- 24GB+ VRAM recommended (RTX 4090 used in experiments)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/cloud-robust-water-detection.git
cd cloud-robust-water-detection

# Create conda environment
conda create -n water-detection python=3.10
conda activate water-detection

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
# requirements.txt
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
rasterio>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
scikit-image>=0.21.0
imageio>=2.31.0
planetary-computer>=0.5.0
pystac-client>=0.6.0
cartopy>=0.21.0
pyproj>=3.5.0
```

---

## 📁 Dataset

### Data Structure

```
data/
├── All_Clear/
│   └── data/
│       ├── train/
│       │   ├── roi100315/
│       │   │   ├── 2022_1/
│       │   │   │   ├── s2_toa/          # Sentinel-2 optical (13 bands)
│       │   │   │   ├── s1/              # Sentinel-1 SAR (VV, VH)
│       │   │   │   ├── dw/              # Dynamic World labels
│       │   │   │   └── cld_shdw/        # Cloud probability mask
│       │   │   ├── 2022_4/
│       │   │   ├── 2022_7/
│       │   │   └── 2022_10/
│       │   ├── roi100316/
│       │   └── ...
│       └── val/
│           └── ...
```

### Sentinel-2 Bands Used

| Index | Band | Wavelength (nm) | Resolution | Description |
|-------|------|-----------------|------------|-------------|
| 0 | B02 | 490 | 10m | Blue |
| 1 | B03 | 560 | 10m | Green |
| 2 | B04 | 665 | 10m | Red |
| 3 | B08 | 842 | 10m | NIR |
| 4 | B11 | 1610 | 20m | SWIR-1 |
| 5 | B12 | 2190 | 20m | SWIR-2 |

### Sentinel-1 Bands Used

| Index | Polarization | Description |
|-------|--------------|-------------|
| 0 | VV | Vertical-Vertical backscatter |
| 1 | VH | Vertical-Horizontal backscatter |

### Data Statistics

| Split | ROIs | Samples | Resolution | Temporal Coverage |
|-------|------|---------|------------|-------------------|
| Train | 3,498 | 13,992 | 256×256 | 4 quarters/year |
| Val | 874 | 3,496 | 256×256 | 4 quarters/year |

---

## 🧩 Model Components

### 1. Siamese UNet Encoder

```python
class SiameseUNetEncoder(nn.Module):
    """
    Shared-weight encoder for multi-temporal processing.
    Each time step is encoded independently, then aggregated.
    
    Input:  (B, T, C, H, W) - T temporal frames
    Output: E4_agg, [E1_agg, E2_agg, E3_agg, E4_agg]
    """
    def __init__(self, in_channels=6, num_frames=4):
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        # Skip connection reduction (T*C -> C)
        self.skip_reduce1 = nn.Conv2d(64*num_frames, 64, 1)
        self.skip_reduce2 = nn.Conv2d(128*num_frames, 128, 1)
        # ...
```

### 2. Prithvi-EO-2.0 Integration

NASA's Prithvi-EO-2.0 is a **600M parameter** foundation model pre-trained on:
- **4.2M+ Harmonized Landsat-Sentinel-2 images**
- **Temporal and spatial attention** for multi-temporal understanding
- **Masked autoencoder** pre-training objective

```python
# Prithvi encoder outputs 1024-dim features
# With CLS token: (B, 785, 1024) = (B, 1 + 14×14×4, 1024)
PRITHVI_DIM = 1024
PRITHVI_WITH_CLS = 785  # 1 CLS + 14×14×4 patches
```

### 3. Attention Fusion Module

```python
class AttentionFusion(nn.Module):
    """
    Learnable attention-based fusion of optical and SAR features.
    Cloud probability serves as a soft hint for modality weighting.
    """
    def forward(self, opt_feat, sar_feat, cloud_prob):
        # Concatenate: [Optical, SAR, CloudHint]
        combined = torch.cat([opt_feat, sar_feat, cloud_prob], dim=1)
        
        # Learn attention weights
        attn_logits = self.attention(combined)  # (B, 2, H, W)
        weights = F.softmax(attn_logits, dim=1)  # [w_opt, w_sar]
        
        # Weighted fusion
        fused = weights[:, 0:1] * opt_feat + weights[:, 1:2] * sar_feat
        return fused, weights
```

---

## 🚀 Usage

### Training Cloud Removal Model

```python
from models import SiameseUPrithvi
from datasets import CloudRemovalDataset

# Initialize model
model = SiameseUPrithvi(
    in_channels=6,
    num_frames=4,
    prithvi_dim=1024
).to('cuda')

# Load dataset
dataset = CloudRemovalDataset(
    data_root='/path/to/All_Clear/data',
    max_samples=5000
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(50):
    for batch in train_loader:
        cloudy = batch['frames'].to('cuda')  # (B, 4, 6, 224, 224)
        clear = batch['gt'].to('cuda')        # (B, 4, 6, 224, 224)
        
        restored = model(cloudy)
        loss = l1_loss(restored, clear) + ssim_loss(restored, clear)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Training Water Detection Model

```python
from models import LateFusionWaterDetection
from datasets import WaterDetectionDataset

# Initialize model
model = LateFusionWaterDetection(
    optical_channels=6,
    sar_channels=2,
    num_classes=1
).to('cuda')

# Training with hybrid loss
criterion = lambda pred, gt: 0.5 * bce_loss(pred, gt) + 0.5 * lovasz_loss(pred, gt)

for epoch in range(30):
    for batch in train_loader:
        s2 = batch['s2'].to('cuda')      # (B, 4, 6, H, W)
        s1 = batch['s1'].to('cuda')      # (B, 4, 2, H, W)
        cloud = batch['cloud'].to('cuda') # (B, 4, 1, H, W)
        gt = batch['gt'].to('cuda')       # (B, 4, H, W)
        
        pred, weights = model(s2, s1, cloud)
        loss = criterion(pred, gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Inference Pipeline

```python
from pipeline import CloudRobustWaterDetection

# Load trained models
pipeline = CloudRobustWaterDetection(
    cloud_removal_ckpt='checkpoints/cloud_removal.pt',
    water_detection_ckpt='checkpoints/water_detection.pt'
)

# Run inference
results = pipeline.predict(
    s2_input,  # (4, 6, 224, 224) - 4 temporal frames
    s1_input,  # (4, 2, 224, 224) - SAR imagery
    verbose=True
)

# Access results
water_mask = results['water_mask']      # (4, 224, 224) - Binary mask
water_prob = results['water_prob']      # (4, 224, 224) - Probability
restored_s2 = results['restored_s2']    # (4, 6, 224, 224) - Cloud-free
sar_weights = results['sar_weights']    # (4, 224, 224) - Attention weights
```

---

## 🔬 Experiments

### 1. Cloud Coverage Experiment (`cloud_coverage_vs_iou_experiment.ipynb`)

Evaluates model robustness across 0-100% synthetic cloud coverage:

```python
# Generate synthetic clouds targeting water bodies
cloud_mask = generate_nonlinear_cloud(
    H, W, water_gt, 
    cloud_coverage=0.5,  # 50% coverage
    seed=42
)

# Apply to clear image
cloudy_s2 = apply_cloud_nonlinear(clear_s2, cloud_mask)

# Compare Model vs MNDWI
model_iou = compute_iou(model_pred, water_gt)
mndwi_iou = compute_iou(mndwi_pred, water_gt)
```

### 2. Real-World Case Study: Pakistan Flood 2022 (`pakistan_flood_mosaic.ipynb`)

Multi-tile mosaic analysis of the devastating 2022 Pakistan floods:

```python
# Study area: Sindh Province
BBOX = {
    'min_lon': 67.5, 'max_lon': 68.5,
    'min_lat': 26.5, 'max_lat': 27.5
}

# Compare pre-flood vs post-flood
pre_flood_date = "2022-06-15"  # Before monsoon
post_flood_date = "2022-09-05"  # Peak flooding
```

### 3. Attention Weight Visualization

The model learns to automatically rely more on SAR in cloudy regions:

```
┌─────────────────────────────────────────────────────────────┐
│              SAR Weight Visualization                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Clear Conditions:        Cloudy Conditions:               │
│   ┌──────────────┐        ┌──────────────┐                 │
│   │ ░░░░░░░░░░░░ │        │ ████████████ │ ← High SAR     │
│   │ ░░░░░░░░░░░░ │        │ ████████████ │   weight       │
│   │ ░░░░░░░░░░░░ │        │ ████████████ │   (>0.8)       │
│   │ ░░░░░░░░░░░░ │        │ ░░░░░░░░░░░░ │ ← Low SAR      │
│   └──────────────┘        └──────────────┘   weight        │
│   SAR Weight: ~0.3        SAR Weight: ~0.8                 │
│                                                             │
│   Legend: ░ = Low weight (0.0-0.4)                         │
│           ▓ = Medium weight (0.4-0.6)                      │
│           █ = High weight (0.6-1.0)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Results Visualization

### West Arnhem, Australia Case Study

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CLOUD REMOVAL RESULTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BEFORE (Cloudy)                    AFTER (Restored)                       │
│   ┌─────────────────────┐            ┌─────────────────────┐               │
│   │     ████████        │            │                     │               │
│   │   ████████████      │            │    ╭─────╮         │               │
│   │  ██████████████     │     ───►   │   ╭╯     ╰╮        │               │
│   │ ████████████████    │            │  ╭╯ RIVER ╰╮       │               │
│   │  ██████████████     │            │   ╰╮     ╭╯        │               │
│   │   ████████████      │            │    ╰─────╯         │               │
│   │     ████████        │            │                     │               │
│   └─────────────────────┘            └─────────────────────┘               │
│        Clouds ██                     Geographical boundary                 │
│                                      successfully restored!                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Saemangeum Dam, Korea (군산 새만금 댐)

Used for cloud coverage experiments with synthetic cloud injection:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│   Cloud 0%    │    Cloud 50%    │    Cloud 100%   │                         │
├───────────────┼─────────────────┼─────────────────┤                         │
│ Model:  0.946 │  Model:  0.941  │  Model:  0.889  │ ← Stable!              │
│ MNDWI:  0.933 │  MNDWI:  0.520  │  MNDWI:  0.000  │ ← Degrades!            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔮 Future Work

### Short-term Improvements

1. **Optical-to-SAR Translation**: Use generative models (e.g., TerraMind, Diffusion) to synthesize SAR from optical imagery, addressing temporal gaps
2. **Near-Daily Monitoring**: Combine Sentinel-1/2 with Landsat and Planet data for daily water body updates
3. **Uncertainty Quantification**: Add MC-Dropout or ensemble methods for prediction confidence

### Long-term Research Directions

1. **Global Water Monitoring System**: Scale to continental/global coverage
2. **Multi-Task Learning**: Joint cloud removal + segmentation + change detection
3. **Foundation Model Fine-tuning**: Explore LoRA, Adapters for efficient domain adaptation
4. **Real-time Processing**: Edge deployment for disaster response

### Known Limitations

| Limitation | Description | Potential Solution |
|------------|-------------|-------------------|
| Temporal Gap | 1-2 day gap between S1/S2 acquisitions | Optical-to-SAR translation |
| Radiometric Accuracy | Restored images are model-generated, not actual observations | Ensemble with confidence scores |
| Computation Cost | Full pipeline requires ~24GB VRAM | Model distillation, quantization |

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2025cloudrobust,
  title={Cloud-Robust Water Body Detection: Fine-Tuning NASA's Prithvi-EO-2.0},
  author={Kim, Beomsik and Kim, Hyunglok},
  booktitle={Environmental Olympiad 2025},
  year={2025},
  organization={GIST Hydro AI Lab}
}
```

### Related Works

```bibtex
@article{jakubik2024prithvi,
  title={Foundation Models for Generalist Geospatial Artificial Intelligence},
  author={Jakubik, Johannes and others},
  journal={arXiv preprint arXiv:2310.18660},
  year={2024}
}

@article{bui2025cloudaware,
  title={Cloud-Aware SAR Fusion for Enhanced Optical Sensing in Space Missions},
  author={Bui, Trong-An},
  year={2025}
}
```

---

## 🙏 Acknowledgments

- **NASA & IBM** for the Prithvi-EO-2.0 foundation model
- **Microsoft Planetary Computer** for satellite data access
- **GIST Hydro AI Lab** for computational resources and guidance
- **ESA Copernicus Programme** for Sentinel-1/2 data

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

- **Beomsik Kim** - beomsik@gist.ac.kr
- **Prof. Hyunglok Kim** - hyunglokkim@gist.ac.kr
- **GIST Hydro AI Lab** - [Lab Website](https://hydroai.gist.ac.kr)

---

<p align="center">
  <b>🌊 Making water monitoring resilient to clouds 🌊</b><br>
  <i>GIST Hydro AI Lab × NASA Prithvi</i>
</p>
