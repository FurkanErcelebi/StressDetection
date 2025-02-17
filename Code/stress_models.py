
# from this site https://github.com/KastoneX/AutoData-Analysis-and-Price-Prediction-of-Used-Cars

import io
import time
import random
import os
import copy
from glob import glob
from tqdm import tqdm

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, f1_score,accuracy_score,recall_score,roc_auc_score,roc_curve, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
# from torchinfo import summary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision
from torchvision import datasets, models, transforms
# from torchmetrics.functional.classification import auroc # auc
from collections import defaultdict

#sns.set(style='whitegrid', palette='muted', font_scale=1.2)

#Colour_Palette = ['#01BEFE', '#FF7D00', '#FFDD00', '#FF006D', '#ADFF02', '#8F00FF']
#sns.set_palette(sns.color_palette(Colour_Palette))

#tqdm.pandas()

import warnings
warnings.simplefilter('ignore')


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_size

        # Number of hidden layers
        self.layer_dim = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):

        ## from this site start from https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#step-3-create-model-class
        
        
# from this site https://github.com/StChenHaoGitHub/1D-deeplearning-model-pytorch/blob/main/AlexNet.py

class AlexNetModifiedVersion(torch.nn.Module):

