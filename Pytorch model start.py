import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

# Reference: https://www.youtube.com/watch?v=tHL5STNJKag
from roboflow import Roboflow
from apiconfig import ROBOFLOW_API_KEY, WORKSPACE_NAME, PROJECT_NAME

rf = Roboflow(api_key=str(ROBOFLOW_API_KEY))
project = rf.workspace(str(WORKSPACE_NAME)).project(str(PROJECT_NAME))
dataset = project.version(4).download("yolov8")
