# MMA3001: Pork Rasher Packaging Defect Detection System

An automated computer-vision quality inspection pipeline for detecting and classifying packaging defects in pork rasher trays using PyTorch and EfficientNet-B0 transfer learning.

---

## 📌 Project Overview

Manual visual inspection on high-speed conveyor lines suffers from operator fatigue, subjective bias, and inconsistency. This project provides a deep learning solution to classify 5 packaging defect categories alongside a clean baseline state:

| Class ID ($c$) | Defect Category | Defect Severity | Operational Action |
| :---: | :--- | :---: | :---: |
| **0** | `loose-meat` | Minor Placement Anomaly | **PASS** (Re-Sort) |
| **1** | `packaging-error` | Critical Structural Defect | **FAIL** (Reject Bin) |
| **2** | `twisted-meat` | Minor Placement Anomaly | **PASS** (Re-Sort) |
| **3** | `unsealed` | Critical Seal Failure | **FAIL** (Reject Bin) |
| **4** | `wrinkle` | Minor Surface Crease | **PASS** (Re-Sort) |
| **-1 (Mapped to 5)** | `no-defect` | Clean Standard Package | **PASS** (Distribution) |

---

## 🛠️ Environment & Dependencies

This codebase requires Python 3.10+ and the following PyTorch ecosystem packages:

```bash
pip install torch torchvision timm roboflow matplotlib pandas numpy tqdm Pillow ipywidgets pytest
```

### Configuration (`apiconfig.py`)
Ensure an `apiconfig.py` file is present in your workspace directory containing your Roboflow credentials:
```python
ROBOFLOW_API_KEY = "your_actual_roboflow_api_key"
WORKSPACE_NAME = "hello-2aqe0"
PROJECT_NAME = "pork-rasher-error-packaging"
```

---

## 📁 Dataset Directory Layout

The dataset is retrieved via the Roboflow API (`Pork-Rasher-Error(Packaging)-4`) and formatted in YOLO annotation structure:

```
Pork-Rasher-Error(Packaging)-4/
├── data.yaml
├── README.dataset.txt
├── README.roboflow.txt
├── train/
│   ├── images/  (*.jpg, *.png)
│   └── labels/  (*.txt -> class_id x_center y_center width height)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 🏗️ Architecture & Component Overview

1. **`PorkRasherPackagingDataset(Dataset)`**:
   - Loads image files and corresponding YOLO text labels (`class_id x_center y_center w h`).
   - Converts annotations to 2D PyTorch tensors reshaped to `(-1, 5)`.
   - Standardizes RGB image tensors to specified dimensions (e.g., $200 \times 200$ or $320 \times 320$).

2. **`collate_fn(batch)`**:
   - Custom batch collation function that stacks images into a 4D tensor `[B, C, H, W]` while keeping label tensors as a variable-length list of tensors to handle images with multiple or zero annotations.

3. **Target Label Mapping**:
   - Non-negative YOLO class IDs `{0, 1, 2, 3, 4}` map to target indices `{0, 1, 2, 3, 4}`.
   - Clean/unannotated images (`NO_DEFECT_CLASS = -1`) are assigned to target index `5`.

4. **`PorkRasherClassifer(nn.Module)`**:
   - Feature Backbone: Pretrained `efficientnet_b0` from `timm`.
   - Classification Head: `nn.Sequential(nn.Flatten(), nn.Linear(1280, num_classes))`.
   - Loss Function: `nn.CrossEntropyLoss()`.
   - Optimizer: `optim.Adam(model.parameters(), lr=0.0001)`.

---

## 🧪 Testing Suite

Unit tests are included to verify dataset parsing, target mapping logic, collate functions, and neural network tensor dimensions.

To run the test suite:
```bash
pytest test_pipeline.py
```
or via unittest:
```bash
python -m unittest test_pipeline.py
```

---

## 📄 License

This project is open-source and released under the **MIT License**. See `LICENSE` for details.
