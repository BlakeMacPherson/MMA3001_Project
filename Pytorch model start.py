# Reference: https://www.youtube.com/watch?v=tHL5STNJKag
from gettext import install
import glob
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
from PIL import Image

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
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
            self.image_paths.extend(glob.glob(os.path.join(self.images_dir, ext)))
        self.image_paths.sort()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_path = os.path.join(
            self.labels_dir,
            os.path.splitext(os.path.basename(image_path))[0] + ".txt",
        )
        with open(label_path, "r", encoding="utf-8") as label_file:
            label_rows = [
                [float(value) for value in line.split()]
                for line in label_file
                if line.strip()
            ]
        label = torch.tensor(label_rows, dtype=torch.float32).reshape(-1, 5)
        return image, label

#images already resized to 640x640, so no need to resize again
transform = transforms.ToTensor()


trainingdataset = PorkRasherPackagingDataset(
    images_dir=os.path.join(dataset_location, "train", "images"),
    labels_dir=os.path.join(dataset_location, "train", "labels"),
    transform=transform
) 


print((trainingdataset.__getitem__(4)))




dataloader = DataLoader(trainingdataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    break

images.shape, labels.shape

"""
class SimpleMeatPackageClassifer(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleMeatPackageClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output
model = SimpleCardClassifer(num_classes=6)
print(str(model)[:500])
"""

#collate fn, stack
