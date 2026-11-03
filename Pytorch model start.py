# Reference: https://www.youtube.com/watch?v=tHL5STNJKag
from gettext import install
import sys
import os
import pip
import roboflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

# Print versions for verification
print('System Version:', sys.version)
print('PyTorch version:', torch.__version__)
print('Torchvision version:', torchvision.__version__)
print('Numpy version:', np.__version__)
print('Pandas version:', pd.__version__)

# Import configuration settings
try:
    from apiconfig import ROBOFLOW_API_KEY, WORKSPACE_NAME, PROJECT_NAME
except ImportError:
    raise ImportError(
        "Could not import from 'apiconfig.py'. "
        "Ensure 'apiconfig.py' exists in your workspace directory."
    )

# Validate API key and configuration
if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "None" or ROBOFLOW_API_KEY == "your_roboflow_api_key_here":
    raise ValueError(
        "Invalid ROBOFLOW_API_KEY detected! "
        "Please open 'apiconfig.py' and set ROBOFLOW_API_KEY to your actual string key "
        "(e.g., ROBOFLOW_API_KEY = 'xyz123...'), not None or 'None'."
    )

from roboflow import Roboflow

print(f"Connecting to Roboflow workspace '{WORKSPACE_NAME}', project '{PROJECT_NAME}'...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)

# Dataset 3 from the MMA3001 brief uses version 4.
dataset_location = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Pork-Rasher-Error(Packaging)-4",
)
dataset_is_downloaded = (
    os.path.isfile(os.path.join(dataset_location, "data.yaml"))
    and os.path.isdir(os.path.join(dataset_location, "train", "images"))
    and os.path.isdir(os.path.join(dataset_location, "valid", "images"))
)
if dataset_is_downloaded:
    print("Using existing dataset at:", dataset_location)
else:
    dataset = project.version(4).download("yolov7", location=dataset_location)
    dataset_location = dataset.location
    print("Dataset downloaded successfully to:", dataset_location)

# Reference: https://www.youtube.com/watch?v=tHL5STNJKag
class PorkRasherPackagingDataset(Dataset):
    def __init__(self, data_location, transform=None):
        self.data = ImageFolder(data_location, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes



print(PorkRasherPackagingDataset(dataset_location).classes)
