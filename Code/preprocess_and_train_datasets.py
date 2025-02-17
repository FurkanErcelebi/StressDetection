# from this site https://github.com/KastoneX/AutoData-Analysis-and-Price-Prediction-of-Used-Cars

import io
import sys
import time
import random
import math
import os
import copy
from glob import glob
from tqdm import tqdm
from datetime import datetime

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, f1_score,accuracy_score,recall_score,roc_auc_score,roc_curve, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision
from torchvision import datasets, models, transforms
from torchmetrics.functional.classification import auroc # auc
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import binary_f1_score, binary_accuracy, binary_precision
from torcheval.metrics.functional.classification import binary_recall

import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, TENSORS

import tensorflow as tf
import tensorboard as tb
from packaging import version

#%matplotlib inline

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

Colour_Palette = ['#01BEFE', '#FF7D00', '#FFDD00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(Colour_Palette))

tqdm.pandas()

import warnings
warnings.simplefilter('ignore')


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
#if torch.cuda.is_available():
#  print(torch.cuda.get_device_name())
  

def get_filtered_columns(df_sample, number_of_features):
    
    le = LabelEncoder()
    df_org = df_sample.copy()
    df_org['Diagnose'] = le.fit_transform(df_org['Diagnose'])
    df_org = df_org.drop('is_augmented', axis=1)
    corr_matrix = df_org.corr()
    corr_with_target = corr_matrix['Diagnose']
    selected_columns = corr_with_target.abs().sort_values(ascending=False)[1:number_of_features].index.values
    return selected_columns


def get_classification_distribution(series_values):

    for unique_target in np.unique(series_values):
      print(unique_target, list(series_values).count(unique_target))


def balance_diagnose_classification_lables(result_all_metric_dataframe, org_class_rate = 1, aug_orig_rate = 4):

  new_metric_dataframe = pd.DataFrame()
  df_reference = result_all_metric_dataframe.copy()

  all_metric_dataframe_with_na_droped_infos_copy = df_reference[df_reference['is_augmented'] == False].copy()
  all_metric_dataframe_with_na_droped_infos_copy_stress = all_metric_dataframe_with_na_droped_infos_copy[all_metric_dataframe_with_na_droped_infos_copy['Diagnose'] == 'Stress']
  aug_number_of_stress = len(all_metric_dataframe_with_na_droped_infos_copy_stress)
  print('Number of "Stress" in original samples :', aug_number_of_stress)
  all_metric_dataframe_with_na_droped_infos_copy_no_stress = all_metric_dataframe_with_na_droped_infos_copy[all_metric_dataframe_with_na_droped_infos_copy['Diagnose'] == 'No Stress']
  aug_number_of_no_stress = len(all_metric_dataframe_with_na_droped_infos_copy_no_stress)
  print('Number of "No Stress" in original samples :',aug_number_of_no_stress)
  if aug_number_of_stress > aug_number_of_no_stress:
    all_metric_dataframe_with_na_droped_infos_stress_undersampled = all_metric_dataframe_with_na_droped_infos_copy_stress.sample(n=aug_number_of_no_stress * org_class_rate, random_state=42, replace=True)
    new_metric_dataframe = pd.concat([new_metric_dataframe, all_metric_dataframe_with_na_droped_infos_stress_undersampled, all_metric_dataframe_with_na_droped_infos_copy_no_stress])
  elif aug_number_of_stress < aug_number_of_no_stress:
    all_metric_dataframe_with_na_droped_infos_no_stress_undersampled = all_metric_dataframe_with_na_droped_infos_copy_no_stress.sample(n=aug_number_of_stress * org_class_rate, random_state=42, replace=True)
    new_metric_dataframe = pd.concat([new_metric_dataframe, all_metric_dataframe_with_na_droped_infos_no_stress_undersampled, all_metric_dataframe_with_na_droped_infos_copy_stress])
  else:
    new_metric_dataframe = pd.concat([new_metric_dataframe, all_metric_dataframe_with_na_droped_infos_copy_stress, all_metric_dataframe_with_na_droped_infos_copy_no_stress])

  number_of_original_sample = len(new_metric_dataframe)
  new_aug_metric_dataframe = pd.DataFrame()


  all_metric_dataframe_with_na_droped_infos_copy = df_reference[df_reference['is_augmented']].copy()
  all_metric_dataframe_with_na_droped_infos_copy_stress = all_metric_dataframe_with_na_droped_infos_copy[all_metric_dataframe_with_na_droped_infos_copy['Diagnose'] == 'Stress']
  aug_number_of_stress = len(all_metric_dataframe_with_na_droped_infos_copy_stress)
  print('Number of "Stress" in augmented samples :', aug_number_of_stress)
  all_metric_dataframe_with_na_droped_infos_copy_no_stress = all_metric_dataframe_with_na_droped_infos_copy[all_metric_dataframe_with_na_droped_infos_copy['Diagnose'] == 'No Stress']
  aug_number_of_no_stress = len(all_metric_dataframe_with_na_droped_infos_copy_no_stress)
  print('Number of "No Stress" in augmented samples :',aug_number_of_no_stress)

  if aug_orig_rate > 0:
    new_aug_sample_size = int((number_of_original_sample * aug_orig_rate) / 2)
    all_metric_dataframe_with_na_droped_infos_stress_undersampled = all_metric_dataframe_with_na_droped_infos_copy_stress.sample(n=new_aug_sample_size, random_state=42, replace=True)
    all_metric_dataframe_with_na_droped_infos_no_stress_undersampled = all_metric_dataframe_with_na_droped_infos_copy_no_stress.sample(n=new_aug_sample_size , random_state=42, replace=True)
    new_aug_metric_dataframe = pd.concat([all_metric_dataframe_with_na_droped_infos_stress_undersampled, all_metric_dataframe_with_na_droped_infos_no_stress_undersampled])
  elif aug_number_of_stress > aug_number_of_no_stress:
    all_metric_dataframe_with_na_droped_infos_stress_undersampled = all_metric_dataframe_with_na_droped_infos_copy_stress.sample(n=aug_number_of_no_stress, random_state=42, replace=True)
    new_aug_metric_dataframe = pd.concat([all_metric_dataframe_with_na_droped_infos_stress_undersampled, all_metric_dataframe_with_na_droped_infos_copy_no_stress])
  elif aug_number_of_stress < aug_number_of_no_stress:
    all_metric_dataframe_with_na_droped_infos_no_stress_undersampled = all_metric_dataframe_with_na_droped_infos_copy_no_stress.sample(n=aug_number_of_stress , random_state=42, replace=True)
    new_aug_metric_dataframe = pd.concat([all_metric_dataframe_with_na_droped_infos_no_stress_undersampled, all_metric_dataframe_with_na_droped_infos_copy_stress])
  else:
    new_aug_metric_dataframe = pd.concat([all_metric_dataframe_with_na_droped_infos_copy_stress, all_metric_dataframe_with_na_droped_infos_copy_no_stress])
  new_metric_dataframe = pd.concat([new_metric_dataframe, new_aug_metric_dataframe])

  return new_metric_dataframe


def get_samples_as_dataloader_for_scaled_with_augmented_datas(input_samples, output_samples, aug_input_samples = None, aug_output_samples = None, is_lstm=False):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    le = OneHotEncoder()
    
    total_input_samples = copy.deepcopy(input_samples)
    
    if aug_input_samples is not None:
        total_input_samples = np.concatenate((total_input_samples, aug_input_samples), axis=0)
    
    scaler.fit(total_input_samples)
    
    total_output_samples = copy.deepcopy(output_samples)
    
    if aug_input_samples is not None:
        total_output_samples = np.concatenate((total_output_samples, aug_output_samples), axis=0)
    
    le.fit(total_output_samples)
    
    scaled_X_total = scaler.transform(input_samples)
    print(input_samples.shape)
    print(scaled_X_total[:5][:5])
    
    scaled_y_total = le.transform(output_samples.reshape(-1, 1))
    print(output_samples[:5])
    print(scaled_y_total.toarray()[:5])
    
    if is_lstm:
        scaled_X_total = scaled_X_total.reshape(scaled_X_total.shape[0], 1, scaled_X_total.shape[1])
    else:
        scaled_X_total = scaled_X_total.reshape(scaled_X_total.shape[0], scaled_X_total.shape[1], 1)
    
    tX_total = torch.tensor(scaled_X_total, dtype=torch.float32)
    ty_total = torch.tensor(scaled_y_total.toarray(), dtype=torch.float32)
    print(tX_total.shape, ty_total.shape)
    
    sample_dataset_total = TensorDataset(tX_total, ty_total)
    
    aug_sample_dataset_total = None
    
    aug_input_samples = None, aug_output_samples
    
    if aug_input_samples is not None and aug_output_samples is not None:
        
        aug_scaled_X_total = scaler.transform(aug_input_samples)
        print(aug_input_samples.shape)
        print(aug_scaled_X_total[:5][:5])
        
        aug_scaled_y_total = le.transform(aug_output_samples.reshape(-1, 1))
        print(aug_output_samples[:5])
        print(aug_scaled_y_total.toarray()[:5])
        
        if is_lstm:
            aug_scaled_X_total = aug_scaled_X_total.reshape(aug_scaled_X_total.shape[0], 1, aug_scaled_X_total.shape[1])
        else:
            aug_scaled_X_total = aug_scaled_X_total.reshape(aug_scaled_X_total.shape[0], saug_caled_X_total.shape[1], 1)
        
        aug_tX_total = torch.tensor(aug_scaled_X_total, dtype=torch.float32)
        aug_ty_total = torch.tensor(aug_scaled_y_total.toarray(), dtype=torch.float32)
        print(aug_tX_total.shape, aug_ty_total.shape)
        
        aug_sample_dataset_total = TensorDataset(aug_tX_total, aug_ty_total)
    
    
    return (sample_dataset_total, aug_sample_dataset_total)


def get_samples_as_dataloader(input_samples, output_samples, is_lstm=False):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    le = OneHotEncoder()
    
    scaled_X_total = scaler.fit_transform(input_samples)
    print(input_samples.shape)
    print(scaled_X_total[:5][:5])
    
    scaled_y_total = le.fit_transform(output_samples.reshape(-1, 1))
    print(output_samples[:5])
    print(scaled_y_total.toarray()[:5])
    
    
    if is_lstm:
        scaled_X_total = scaled_X_total.reshape(scaled_X_total.shape[0], 1, scaled_X_total.shape[1])
    else:
        scaled_X_total = scaled_X_total.reshape(scaled_X_total.shape[0], scaled_X_total.shape[1], 1)
    
    tX_total = torch.tensor(scaled_X_total, dtype=torch.float32)
    ty_total = torch.tensor(scaled_y_total.toarray(), dtype=torch.float32)
    print(tX_total.shape, ty_total.shape)
    
    sample_dataset_total = TensorDataset(tX_total, ty_total)
    
    return sample_dataset_total
    

def create_and_get_log_writer(model_name):
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    experiment_name = "StressDetection"
    log_dir = os.path.join("run",timestamp, experiment_name, model_name)
    log_writer = SummaryWriter(log_dir)
    
    return log_writer
    

def get_detaild_model_infos(selected_model, model_input_size):
    
    first_parameter = next(selected_model.parameters())
    print(first_parameter.shape)
    print(summary(model=selected_model, input_size=model_input_size, col_width=20,
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        row_settings=['var_names'], verbose=0))
    

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def train_model(dataset_sizes, dataloaders, selected_time_series_datasets, selected_model , selected_device, criterion, optimizer, scheduler, log_writer, name, batch_size, num_epochs=25,k_cross_fold_idx=5, aug_dataloader = None):

    #Creating a folder to save the model performance.
    try:
        os.mkdir(f'./modelPerformance/')
    except:
        print('Dosya var')

    model_performance_path = f'./modelPerformance/{name}'
    try:
        os.mkdir(model_performance_path)
    except:
        print('Dosya var')

    since = time.time()

    best_model_wts = copy.deepcopy(selected_model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        phase_metrics = { x: { 'loss': 0, 'acc': 0}  for x in ['train', 'val']}
        phase_metrics['val'] = { x : 0  for x in ['f1score', 'precision', 'recall']}
        for phase in ['train', 'val']:
            if phase == 'train':
                selected_model.train()  # Set model to training mode
            else:
                selected_model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            #epochs

            number_of_batch = int(len(selected_time_series_datasets[phase])/batch_size)

            total_number_of_batch = number_of_batch

            if phase == 'train':
              aug_number_of_batch = int(len(selected_time_series_datasets['train_aug'])/batch_size)
              total_number_of_batch = total_number_of_batch + aug_number_of_batch
            
            y_pred = []
            y_true = []
            for batch_index in tqdm(range(total_number_of_batch),desc=f'Training process for {phase} phase'):
                #Loading Data

                if phase == 'train' and batch_index > number_of_batch and aug_dataloader is not None:
                  inputs, labels = next(iter(aug_dataloader))
                else:
                  inputs, labels = next(iter(dataloaders[phase]))

                inputs = inputs.to(selected_device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = selected_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    preds_output = (preds).data.cpu().numpy()
                    y_pred.extend(preds_output) # Save Prediction

                    labels = labels.to(selected_device)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Convert labels to class indices:
                labels_indices = torch.argmax(labels, dim=1)  # Get indices from one-hot labels
                labels_np = labels_indices.data.cpu().numpy()  # Convert to NumPy for y_true
                y_true.extend(labels_np) # Save Truth
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels_indices)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1score = 0
            epoch_recall = 0
            epoch_precision = 0

            phase_metrics[phase]['loss'] = epoch_loss
            phase_metrics[phase]['acc'] = epoch_acc
            
            #AUC: {:.4f} , epoch_auc
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            
            # deep copy the model
            if phase == 'val':
                
                ty_pred = torch.tensor(y_pred)
                ty_true = torch.tensor(y_true)
                
                epoch_f1score = binary_f1_score(ty_pred, ty_true)
                epoch_recall = binary_recall(ty_pred, ty_true)
                epoch_precision = binary_precision(ty_pred, ty_true)
                print('{} F1 Score: {:.4f} Precision: {:.4f} Recall: {:.4f}'.format(
                    phase, epoch_f1score, epoch_precision, epoch_recall))
                
                phase_metrics[phase]['f1score'] = epoch_f1score
                phase_metrics[phase]['recall'] = epoch_recall
                phase_metrics[phase]['precision'] = epoch_precision
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(selected_model.state_dict())
                    torch.save(selected_model,'{}/best_model_{:.4f}acc_{}epochs_{}Kfold.h5'.format(model_performance_path,epoch_acc,epoch+1, k_cross_fold_idx))

                # train_losses = []
                # valid_losses = []


            time.sleep(2)


        log_writer.add_scalars(main_tag=f'Loss - {k_cross_fold_idx}', tag_scalar_dict={"train/loss": phase_metrics['train']['loss'],
                                                            "val/loss": phase_metrics['val']['loss']}, global_step=epoch)
        log_writer.add_scalars(main_tag=f'Accuracy - {k_cross_fold_idx}', tag_scalar_dict={"train/acc": phase_metrics['train']['acc'],
                                                                  "val/acc": phase_metrics['val']['acc']}, global_step=epoch)
        log_writer.add_scalars(main_tag=f'Others - {k_cross_fold_idx}', tag_scalar_dict={"val/f1score": phase_metrics['val']['f1score'],
                                                                  "val/precision": phase_metrics['val']['precision'],
                                                                  "val/recall": phase_metrics['val']['recall']}, global_step=epoch)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    with open(f'{model_performance_path}/'+sorted(os.listdir(f'{model_performance_path}/'))[-1], 'rb') as f:
        buffer = io.BytesIO(f.read())
        selected_model=torch.load(buffer)
        # load best model weights
        selected_model.load_state_dict(best_model_wts)

    return selected_model


def train_model_in_folds(model_name, selected_device, number_of_splits, selected_num_epochs, selected_batch_size, selected_lr, selected_dataset_total, log_writer,selected_model, selected_loss_fn, selected_optimized, selected_aug_dataset_total = None):
    
    number_of_folder = 0
    selected_model.apply(reset_weights)
    
    while number_of_folder < 5:

      kfold = KFold(n_splits=number_of_splits, shuffle=True)
      for train_ids, test_ids in kfold.split(selected_dataset_total):
    
          number_of_folder += 1
    
          if number_of_folder > 5:
            break
    
          # Print
          print(f'FOLD {number_of_folder}')
          print('--------------------------------')
    
          print(f'TRAIN ids: {len(train_ids)}')
          print(f'TEST ids: {len(test_ids)}')
    
          train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
          test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
          # Define data loaders for training and testing data in this fold
          dataloaders = {
            'train': torch.utils.data.DataLoader(
                              selected_dataset_total,
                              batch_size=selected_batch_size, sampler=train_subsampler),
            'val': torch.utils.data.DataLoader(
                              selected_dataset_total,
                              batch_size=selected_batch_size, sampler=test_subsampler)
    
          }
          
          time_series_datasets = {
              'train': train_subsampler,
              'val': test_subsampler,
          }
            
          selected_aug_dataset_loader = None
          if selected_aug_dataset_total is not None:
            selected_aug_dataset_loader = torch.utils.data.DataLoader(selected_aug_dataset_total, batch_size=selected_batch_size)
            time_series_datasets['train_aug'] = selected_aug_dataset_total.tensors[0]
          else:
            time_series_datasets['train_aug'] = []
          
          dataset_sizes = {x: (len(time_series_datasets[x]) + len(time_series_datasets['train_aug'])) if x == 'train' else len(time_series_datasets[x])
                                for x in ['train', 'val']}
          
          selected_model.apply(reset_weights)
          optimizer = selected_optimized(selected_model.parameters(), lr=selected_lr)  # Learning rate
    
    
          
          exp_lr_scheduler = None
          model_ft = train_model(dataset_sizes, dataloaders, time_series_datasets, selected_model, selected_device, selected_loss_fn, optimizer, exp_lr_scheduler, log_writer, model_name, selected_batch_size, selected_num_epochs, number_of_folder, selected_aug_dataset_loader)
          model_ft.apply(reset_weights)
          


def train_model_in_strified_folds(model_name, selected_device, number_of_splits, selected_num_epochs, selected_batch_size, selected_lr, selected_raw_values, log_writer,selected_model, selected_loss_fn, selected_optimized, selected_aug_dataset_total = None, is_lstm=False):
    
    number_of_folder = 0
    selected_model.apply(reset_weights)
    
    selected_dataset_total = get_samples_as_dataloader(selected_raw_values[0], selected_raw_values[1], is_lstm)
    while number_of_folder < 5:

      kfold = StratifiedKFold(n_splits=number_of_splits, shuffle=True)
      for train_ids, test_ids in kfold.split(selected_raw_values[0], selected_raw_values[1]):
    
          number_of_folder += 1
    
          if number_of_folder > 5:
            break
    
          # Print
          print(f'FOLD {number_of_folder}')
          print('--------------------------------')
    
          print(f'TRAIN ids: {len(train_ids)}')
          print(f'TEST ids: {len(test_ids)}')
    
          train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
          test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
          # Define data loaders for training and testing data in this fold
          dataloaders = {
            'train': torch.utils.data.DataLoader(
                              selected_dataset_total,
                              batch_size=selected_batch_size, sampler=train_subsampler),
            'val': torch.utils.data.DataLoader(
                              selected_dataset_total,
                              batch_size=selected_batch_size, sampler=test_subsampler)
    
          }
          
          time_series_datasets = {
              'train': train_subsampler,
              'val': test_subsampler,
          }
            
          selected_aug_dataset_loader = None
          if selected_aug_dataset_total is not None:
            selected_aug_dataset_loader = torch.utils.data.DataLoader(selected_aug_dataset_total, batch_size=selected_batch_size)
            time_series_datasets['train_aug'] = selected_aug_dataset_total.tensors[0]
          else:
            time_series_datasets['train_aug'] = []
          
          dataset_sizes = {x: (len(time_series_datasets[x]) + len(time_series_datasets['train_aug'])) if x == 'train' else len(time_series_datasets[x])
                                for x in ['train', 'val']}
          
          selected_model.apply(reset_weights)
          optimizer = selected_optimized(selected_model.parameters(), lr=selected_lr)  # Learning rate
    
    
          # Decay LR by a factor of 0.1 every 7 epochs
          exp_lr_scheduler = None#lr_scheduler.StepLR(optimizer, step_size=int(BATCH_SIZE / 5), gamma=0.5)
          # TRAINING
          model_ft = train_model(dataset_sizes, dataloaders, time_series_datasets, selected_model, selected_device, selected_loss_fn, optimizer, exp_lr_scheduler, log_writer, model_name, selected_batch_size, selected_num_epochs, number_of_folder, selected_aug_dataset_loader)
          model_ft.apply(reset_weights)
          



