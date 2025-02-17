

#%load_ext autoreload
#%autoreload 2

import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import os
import random
import re
import datetime
import math
import seaborn as sns
from scipy import stats, signal
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.impute import KNNImputer

#from tensorflow import keras
from IPython.display import SVG, display, Image

def get_labels_df(data_infos, referenced_main_path, expert_column=None):
  sample_path = data_infos['diagnose_result']
  sample_path += '/' + data_infos['sample_name']
  sample_path += '/' + data_infos['game_name']
  #experLabel_df = pd.read_csv(f'{MAIN_PATH}/{sample_path}/ExpertLabels.csv')
  experLabel_df = pd.read_csv(f'{referenced_main_path}/{sample_path}/ExpertLabels.csv')
  number_of_labels = 0
  for label_values in experLabel_df[['Expert1','Expert2','Expert3']].values[1:]:
    if np.any([(label_value.__class__.__name__ == 'str') for label_value in label_values]):
      number_of_labels += 1

  # print(number_of_labels)
  experLabel_df = experLabel_df[['Expert1','Expert2','Expert3']][1: number_of_labels]
  #if expert_column is None:
  diagnose_result_list = []
  for idx in range(0, experLabel_df.shape[0]):
    diagnose_dict = dict(Counter(experLabel_df[['Expert1','Expert2','Expert3']].iloc[idx].values))
    diagnose_result = max(diagnose_dict, key= lambda x: diagnose_dict[x])
    # print(diagnose_result)
    if diagnose_result.__class__.__name__ == 'str':
      diagnose_result_list.append(diagnose_result)

    experLabel_df['Diagnose'] = pd.Series(data=diagnose_result_list)
  #else:
  #  experLabel_df['Diagnose'] = experLabel_df[expert_column]

  experLabel_df = experLabel_df.drop(columns=['Expert1','Expert2','Expert3'], axis=1)
  experLabel_df = experLabel_df.dropna()
  return experLabel_df
  

# from there https://www.geeksforgeeks.org/python-list-all-files-in-directory-and-subdirectories/
def list_files_scandir(allowed_file_list, return_data_list, path='.', refrecened_main_path='/', add_sample_file_name=False):
    with os.scandir(path) as entries:
        entries_copy = list(entries)
        entry_file_paths = [entry.name for entry in entries_copy if entry.is_file()]
        entry_file_path_list = [allowed_file for allowed_file in allowed_file_list if allowed_file in entry_file_paths]
        entry_dir_path_list = [entry.path for entry in entries_copy if entry.is_dir()]
        
        
        if len(entry_file_path_list)  == len(allowed_file_list):
            sample_components = path.split(refrecened_main_path)[1].split('/')
            if add_sample_file_name : 
                for entry_file_path in entry_file_path_list:
                    return_data_list.append({
                        'diagnose_result': sample_components[1],
                        'sample_name': sample_components[2],
                        'game_name': sample_components[3],
                        'file_name': entry_file_path,
                    })
            else:
                return_data_list.append({
                      'diagnose_result': sample_components[1],
                      'sample_name': sample_components[2],
                      'game_name': sample_components[3],
                })
              
        elif len(entry_file_paths) > 0:
          print('Folleing files not found in "{path}" : ', allowed_file_list)

        for entry_dir in entry_dir_path_list:
          list_files_scandir(allowed_file_list, return_data_list, entry_dir, refrecened_main_path, add_sample_file_name)


def get_formatted_values(original_value):
  result_value = original_value
  if type(original_value).__name__ == 'str':
    if len(result_value.split(',')) == 3:
      result_value_parts = result_value.split(',')
      result_value = f'{result_value_parts[0]},{result_value_parts[1]}'

    result_value = result_value.replace(',','.')
  return result_value


def format_time_str(time_value):
  fomatted_time = time_value
  if re.search(r'[a-zA-Z]', fomatted_time.replace(',', '')) is not None:
    return pd.NA

  if len(fomatted_time.split(',')) == 3:
    fomatted_time_parts = fomatted_time.split(',')
    fomatted_time = f'{fomatted_time_parts[0]},{fomatted_time_parts[1]}'
  elif len(fomatted_time.split(',')) == 1:
    fomatted_time = f'{fomatted_time},0000'

  return fomatted_time.replace(',','.')


def get_sample_path(data_infos, dt_procedure, refrecened_main_path='/'):
  sample_path = refrecened_main_path + ' ' + dt_procedure + '/' + data_infos['diagnose_result']
  sample_path += '/' + data_infos['sample_name']
  sample_path += '/' + data_infos['game_name']
  sample_path += '/' + data_infos['file_name']
  return sample_path
  
def read_datas_from_csv(data_infos, dt_procedure, refrecened_main_path='/'):
  sample_path = get_sample_path(data_infos, dt_procedure, refrecened_main_path)
  df = pd.read_csv(sample_path)
  if 'ST' in df.columns:
    df = df.drop(columns=['ST'], axis=1)
  df['values'] = df['values'].apply(lambda x: get_formatted_values(x)).astype(np.float64)
  df['time'] = pd.to_datetime(df['time'].str.replace(',', '.'), unit='s')
  df.index_col = 'time'
  df.index = df['time']
  return df


def write_features_in_csv(data_infos, raw_df, interval_in_seconds, dt_procedure, column_prefix, refrecened_main_path='/', delete_sample=False, special_features = [], get_special_metrics_callback = lambda: None, sampling_rate = 4):
  sample_path = get_sample_path(data_infos, dt_procedure, refrecened_main_path)
  
  sample_path = sample_path.split('.csv')[0] + f'_Features_{interval_in_seconds}s.csv'
  if os.path.exists(sample_path):
    if delete_sample:
      os.remove(sample_path)
      metrics_dataframe = get_signal_metrics(raw_df, interval_in_seconds, column_prefix, [], special_features, get_special_metrics_callback, sampling_rate)#, label_df['Diagnose'].values, data_infos['use_only_stress_labeled_sample'])
      if metrics_dataframe is not None:
        metrics_dataframe.to_csv(sample_path, index=True)
        print(f'{sample_path} file recreated')
    else:
      print(f'{sample_path} file already exist')
  else:
    metrics_dataframe = get_signal_metrics(raw_df, interval_in_seconds, column_prefix, [], special_features, get_special_metrics_callback, sampling_rate)#, label_df['Diagnose'].values, data_infos['use_only_stress_labeled_sample'])
    if metrics_dataframe is not None:
      metrics_dataframe.to_csv(sample_path, index=True)
      print(f'{sample_path} file created')


def get_signal_metrics(df_sample, interval_in_seconds, column_prefix, lable_list = [], special_features = [], get_special_metrics_callback = lambda: None, sampling_rate = 4, get_only_stess_window = False):

  start_time = df_sample['time'].values[0]
  is_remain_row = True
  statistic_value_list = []
  second_count = interval_in_seconds
  feature_columns = ['Seconds'
                    ,f'{column_prefix}_Mean'
                    ,f'{column_prefix}_Median'
                    ,f'{column_prefix}_Min'
                    ,f'{column_prefix}_Max'
                    ,f'{column_prefix}_Kurt'
                    ,f'{column_prefix}_Skew'
                    ,f'{column_prefix}_Q05'
                    ,f'{column_prefix}_Q25'
                    ,f'{column_prefix}_Q75'
                    ,f'{column_prefix}_Q95'
                    ,f'{column_prefix}_Var']
    
  if len(special_features) > 0:
    feature_columns.extend(special_features)    

  idx = 0
  while is_remain_row:
    end_time = start_time + np.timedelta64(interval_in_seconds * 1000, 'ms')
    df_sample.loc[start_time:end_time, 'values']
    df_values = df_sample.loc[start_time:end_time, 'values']
    df_values_array = df_values.values
    if len(df_values_array) > 0:
      statistic_values = []
      if get_only_stess_window and idx < len(lable_list):
        if lable_list[idx] == 'No Stress':
          second_count += interval_in_seconds
          statistic_values = np.full((len(feature_columns),), np.nan)
          statistic_value_list.append(statistic_values)

      if len(statistic_values) == 0:
        statistic_values.append(second_count)
        second_count += interval_in_seconds
        statistic_values.append(df_values_array.mean())
        statistic_values.append(np.median(df_values_array))
        statistic_values.append(df_values_array.min())
        statistic_values.append(df_values_array.max())
        statistic_values.append(stats.kurtosis(df_values_array, fisher=True))
        statistic_values.append(stats.skew(df_values_array))
        statistic_values.append(np.quantile(df_values_array, 0.05))
        statistic_values.append(np.quantile(df_values_array, 0.25))
        statistic_values.append(np.quantile(df_values_array, 0.75))
        statistic_values.append(np.quantile(df_values_array, 0.95))
        statistic_values.append(np.var(df_values_array))
        if len(special_features) > 0:
            tmp_sampling_rate = sampling_rate if len(df_values_array) >= sampling_rate else len(df_values_array)
            special_values_infos = get_special_metrics_callback(df_values, tmp_sampling_rate)
            for special_values_infos_key in special_values_infos.keys():
                statistic_values.append(special_values_infos[special_values_infos_key])
        
        statistic_value_list.append(statistic_values)

      start_time = end_time
    else :
      is_remain_row = False

    idx += 1

  return pd.DataFrame(statistic_value_list, columns = feature_columns)



