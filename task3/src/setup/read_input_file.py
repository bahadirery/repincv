import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utilities import utils

def dataset_based_changes(config_params):
    if config_params['dataset'] == 'cbis-ddsm':
        views_col = 'Views'
        sanity_check_mil_col = 'FolderName'
        sanity_check_sil_col = 'ImageName'
    return views_col, sanity_check_mil_col, sanity_check_sil_col

def input_file_creation(config_params):
    views_col, sanity_check_mil_col, sanity_check_sil_col = dataset_based_changes(config_params)
    if config_params['learningtype'] == 'SIL':
        if config_params['datasplit'] == 'officialtestset':
            csv_file_path = config_params['SIL_csvfilepath']
            df_modality = pd.read_csv(csv_file_path, sep=';')
            print("df modality shape:",df_modality.shape)
            df_modality = df_modality[~df_modality['Views'].isnull()]
            print("df modality no null view:",df_modality.shape)
            df_modality['FullPath'] = config_params['preprocessed_imagepath']+'/'+df_modality['ShortPath']
            if config_params['labeltouse'] == 'imagelabel':
                df_modality['Groundtruth'] = df_modality['ImageLabel']
            elif config_params['labeltouse'] == 'caselabel':
                df_modality['Groundtruth'] = df_modality['CaseLabel']
            
            if config_params['dataset'] == 'cbis-ddsm':
                df_train = df_modality[df_modality['ImageName'].str.contains('Training')]
                if config_params['usevalidation']:
                    df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True, stratify=df_train['Groundtruth'])
                df_test = df_modality[df_modality['ImageName'].str.contains('Test')]
           
            elif config_params['dataset'] == 'vindr':
                df_train = df_modality[df_modality['Split'] == 'training']
                if config_params['usevalidation']:
                    df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True, stratify=df_train['Groundtruth'])
                df_test = df_modality[df_modality['Split'] == 'test']
            total_instances = df_modality.shape[0]

    print("Total instances:", total_instances)
    
    #df_train = df_train[100:140]
    #df_val = df_val[:20]
    #df_test = df_test[20:40]

    #reset index     
    df_train = df_train.reset_index()
    train_instances = df_train.shape[0]
    print("Train:", utils.stratified_class_count(df_train))
    print("training instances:", train_instances)
    if config_params['usevalidation']:
        df_val = df_val.reset_index()
        val_instances = df_val.shape[0]
        print("Val:",utils.stratified_class_count(df_val))
        print("Validation instances:", val_instances)
    df_test = df_test.reset_index()
    test_instances = df_test.shape[0]
    print("Test:", utils.stratified_class_count(df_test)) 
    print("Test instances:", test_instances) 
            
    view_group_indices = None
        
    numbatches_train = int(math.ceil(train_instances/config_params['batchsize']))
    
    if config_params['usevalidation']:
        numbatches_val = int(math.ceil(val_instances/config_params['batchsize']))
    
    numbatches_test = int(math.ceil(test_instances/config_params['batchsize']))
    
    if config_params['usevalidation']:
        return df_train, df_val, df_test, numbatches_train, numbatches_val, numbatches_test, view_group_indices
    else:
        return df_train, df_test, numbatches_train, numbatches_test, view_group_indices