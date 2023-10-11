# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:43:24 2021

@author: PathakS
"""

import re
import os
import math
import time
import torch
import datetime
import argparse
import random
import yaml
import wandb
WANDB__SERVICE_WAIT=500
import sys
import logging

#import timm
import glob
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss

from train_eval import test, optimization, loss_function, evaluation, data_loader
# from models import sil_mil_model, wu_resnet
from models import sil_mil_model
from utilities import pytorchtools, utils, logging_lib
from setup import read_config_file, read_input_file, output_files_setup


## TODO not used delete later
def set_random_seed(config_params):
    #random state initialization of the code - values - 8, 24, 30
    torch.manual_seed(config_params['randseedother']) 
    torch.cuda.manual_seed(config_params['randseedother'])
    torch.cuda.manual_seed_all(config_params['randseedother'])
    np.random.seed(config_params['randseeddata'])
    random.seed(config_params['randseeddata'])
    g = torch.Generator()
    g.manual_seed(config_params['randseedother'])
    torch.backends.cudnn.deterministic = True
    return g

def model_initialization(config_params):
    if config_params['learningtype'] == 'SIL':
        model = sil_mil_model.SILmodel(config_params)
        #model = timm.create_model('tf_efficientnetv2_s', num_classes=1, pretrained=True)
        #num_of_features = model.classifier.in_features
        #model.classifier = nn.Linear(num_of_features, 1)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    #print(model)
    #if config_params['device']=='cuda':
        #cuda_device_list=list(map(int, config_params['device'].split(':')[1].split(',')))
        #model = nn.DataParallel(model, device_ids = [0,1])
    model.to(torch.device(config_params['device']))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total model parameters:", pytorch_total_params, flush=True)

    return model, pytorch_total_params

def model_checkpoint(config_params, path_to_model):
    if config_params['patienceepochs']:
        modelcheckpoint = pytorchtools.EarlyStopping(path_to_model=path_to_model, early_stopping_criteria=config_params['early_stopping_criteria'], patience=config_params['patienceepochs'], verbose=True)
    elif config_params['usevalidation']:
        modelcheckpoint = pytorchtools.ModelCheckpoint(path_to_model=path_to_model, verbose=True)
    return modelcheckpoint

def train(config_params, model, path_to_model, data_iterator_train, data_iterator_val, batches_train, batches_val, df_train, cfg, config_filename):
    '''Training'''
    if config_params['usevalidation']:
        modelcheckpoint = model_checkpoint(config_params, path_to_model)
    optimizer = optimization.optimizer_fn(cfg, model) # Modify it so that it can be used for other optimizers from yaml file
    scheduler = optimization.select_lr_scheduler(config_params, optimizer)
    class_weights_train = loss_function.class_imbalance(config_params, df_train)

    if os.path.isfile(path_to_model):
        model, optimizer, start_epoch = utils.load_model(model, optimizer, path_to_model)
        if config_params['patienceepochs']:
            modelcheckpoint = pytorchtools.EarlyStopping(path_to_model=path_to_model, best_score=config_params['valloss_resumetrain'], early_stopping_criteria=config_params['early_stopping_criteria'], patience=config_params['patienceepochs'], verbose=True)
        print("start epoch:",start_epoch)
        print("lr:",optimizer.param_groups[0]['lr'])
    else:
        start_epoch = 0
        
    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss, bceloss = loss_function.loss_fn_gmic_initialize(config_params, class_weights_train, test_bool=False)
    else:
        if config_params['activation'] == 'softmax':
            lossfn = loss_function.loss_fn_crossentropy(config_params, class_weights_train, test_bool=False)
        elif config_params['activation'] == 'sigmoid':
            lossfn = loss_function.loss_fn_bce(config_params, class_weights_train, test_bool=False)
    
    logging_lib.log_model_info(model)

    for epoch in range(start_epoch,cfg['maxepochs']):
        model.train()
        loss_train=0.0
        correct_train=0
        conf_mat_train=np.zeros((config_params['numclasses'],config_params['numclasses']))
        total_images_train=0
        batch_no=0
        eval_mode = False

        logging_lib.log_gpu_info(epoch)
        
        for train_idx, train_batch, train_labels, views_names in data_iterator_train:
            print('Current Time after one batch loading:', time.ctime(time.time()), flush = True)
            train_batch = train_batch.to(config_params['device'])
            train_labels = train_labels.to(config_params['device'])
            train_labels = train_labels.view(-1)
            print("train labels:", train_labels, flush=True)
            print("train batch:", train_batch.shape, flush=True)
            
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _ = model(train_batch, eval_mode) # compute model output, loss and total train loss over one epoch
                    output_patch = None
                
                if config_params['activation'] == 'sigmoid':
                    output_batch_local = output_batch_local.view(-1)
                    output_batch_global = output_batch_global.view(-1)
                    output_batch_fusion = output_batch_fusion.view(-1)
                    train_labels = train_labels.float()
                    pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
                
                elif config_params['activation'] == 'softmax':
                    pred = output_batch_fusion.argmax(dim=1, keepdim=True)
                loss = loss_function.loss_fn_gmic(config_params, bcelogitloss, bceloss, output_batch_local, output_batch_global, output_batch_fusion, saliency_map, train_labels, class_weights_train, output_patch, test_bool=False)

            else:
                if config_params['learningtype'] == 'SIL':
                    output_batch = model(train_batch, eval_mode)
                    #output_batch = model(train_batch)
                    #print("output batch:", output_batch, flush=True)
                
                if config_params['activation'] == 'sigmoid':
                    if len(output_batch.shape)>1:
                        output_batch = output_batch.squeeze(1)
                    output_batch = output_batch.view(-1)                                                                    
                    train_labels = train_labels.float()
                    pred = torch.ge(torch.sigmoid(output_batch), torch.tensor(0.5)).float()
                    if config_params['classimbalance'] == 'focalloss':
                        loss = sigmoid_focal_loss(output_batch, train_labels, alpha=-1, reduction='mean')
                    else:
                        loss = lossfn(output_batch, train_labels)
                
                elif config_params['activation'] == 'softmax':
                    pred = output_batch.argmax(dim=1, keepdim=True)
                    loss = lossfn(output_batch, train_labels)

            loss_train+=(train_labels.size()[0]*loss.item())

            optimizer.zero_grad()  # clear previous gradients, compute gradients of all variables wrt loss
            loss.backward()
            optimizer.step() # performs updates using calculated gradients
            batch_no=batch_no+1

            #performance metrics of training dataset
            correct_train, total_images_train, conf_mat_train, _ = evaluation.conf_mat_create(pred, train_labels, correct_train, total_images_train, conf_mat_train, config_params['classes'])
            print('Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config_params['maxepochs'], batch_no, batches_train, loss.item()), flush = True)
            print('Current Time after one batch training:', time.ctime(time.time()), flush=True)
        
        if scheduler!=None:
            current_lr=scheduler.get_last_lr()[0]
        else:
            current_lr=optimizer.param_groups[0]['lr']
        print("current lr:",current_lr, flush=True)
        
        running_train_loss = loss_train/total_images_train

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        
        if config_params['usevalidation']:
            correct_test, total_images_val, loss_val, conf_mat_val, auc_val = validation(config_params, model, data_iterator_val, batches_val, df_val, epoch)
            valid_loss = loss_val/total_images_val
            results_dict = evaluation.results_store_excel(True, True, False, None, correct_train, total_images_train, loss_train, correct_test, total_images_val, loss_val, epoch, conf_mat_train, conf_mat_val, current_lr, auc_val, path_to_results_xlsx, path_to_results_text)

            # Log results to wandb that we have in each epoch
            wandb.log(results_dict, step=epoch+1)
        if config_params['patienceepochs']:
            modelcheckpoint(valid_loss, model, optimizer, epoch, conf_mat_train, conf_mat_val, running_train_loss, auc_val)
            if modelcheckpoint.early_stop:
                print("Early stopping",epoch+1, flush = True)
                break
        else:
            if config_params['usevalidation']:
                modelcheckpoint(valid_loss, model, optimizer, epoch, conf_mat_train, conf_mat_val, running_train_loss, auc_val)
            else:
                # Only enters here at the end of the last epoch
                utils.save_model(model, optimizer, epoch, running_train_loss, path_to_model)
                per_model_metrics, conf_mat_test = test(config_params, model, path_to_model, data_iterator_val, batches_val, df_test)
                results_dic = evaluation.results_store_excel(True, False, True, per_model_metrics, correct_train, total_images_train, loss_train, None, None, None, epoch, conf_mat_train, None, current_lr, None, path_to_results_xlsx, path_to_results_text)
                evaluation.write_results_xlsx_confmat(config_params, conf_mat_test, path_to_results_xlsx, 'confmat_train_val_test')
                evaluation.write_results_xlsx(per_model_metrics, path_to_results_xlsx, 'test_results')

        if scheduler!=None: 
            scheduler.step()

        print('Current Time after validation check on the last epoch:', time.ctime(time.time()), flush=True)

        checkpoint_dir = os.path.join(cfg['OUTPUT_BASE'], 'weights', config_filename)
        os.makedirs(checkpoint_dir, exist_ok=True)

        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }

        # save weights at each third epoch
        if epoch % 3 == 0:
        
           checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
           torch.save(state, checkpoint_path)
           logging.info(f'Checkpoint saved to {checkpoint_path}')

    if config_params['usevalidation']:
        evaluation.write_results_xlsx_confmat(config_params, modelcheckpoint.conf_mat_train_best, path_to_results_xlsx, 'confmat_train_val_test')
        evaluation.write_results_xlsx_confmat(config_params, modelcheckpoint.conf_mat_test_best, path_to_results_xlsx, 'confmat_train_val_test')



    print('Finished Training')

    
    
def validation(config_params, model, data_iterator_val, batches_val, df_val, epoch):
    """Validation"""
    model.eval()
    total_images=0
    val_loss = 0
    correct = 0
    s=0
    batch_val_no=0
    eval_mode = True
    conf_mat_val=np.zeros((config_params['numclasses'],config_params['numclasses']))

    class_weights_val = loss_function.class_imbalance(config_params, df_val)

    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss_val, bceloss_val = loss_function.loss_fn_gmic_initialize(config_params, class_weights_val, test_bool=False)
    else:
        if config_params['activation'] == 'softmax':
            lossfn1 = loss_function.loss_fn_crossentropy(config_params, class_weights_val, test_bool=False)
        elif config_params['activation'] == 'sigmoid':
            lossfn1 = loss_function.loss_fn_bce(config_params, class_weights_val, test_bool=False)
    
    with torch.no_grad():   
        for val_idx, val_batch, val_labels, views_names in data_iterator_val:
            val_batch, val_labels = val_batch.to(config_params['device']), val_labels.to(config_params['device'])
            val_labels = val_labels.view(-1)#.float()
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val, _, _, _, _ = model(val_batch, eval_mode) # compute model output, loss and total train loss over one epoch
                    output_patch_val = None
                
                if config_params['activation'] == 'sigmoid':
                    output_batch_local_val = output_batch_local_val.view(-1)
                    output_batch_global_val = output_batch_global_val.view(-1)
                    output_batch_fusion_val = output_batch_fusion_val.view(-1)
                    val_labels = val_labels.float()
                    val_pred = torch.ge(torch.sigmoid(output_batch_fusion_val), torch.tensor(0.5)).float()
                
                elif config_params['activation'] == 'softmax':
                    val_pred = output_batch_fusion_val.argmax(dim=1, keepdim=True)

                loss1 = loss_function.loss_fn_gmic(config_params, bcelogitloss_val, bceloss_val, output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val, val_labels, class_weights_val, output_patch_val, test_bool=False).item()
                output_val = output_batch_fusion_val
            else:
                if config_params['learningtype'] == 'SIL':
                    output_val = model(val_batch, eval_mode)
                
                if config_params['activation'] == 'sigmoid':
                    if len(output_val.shape)>1:
                        output_val = output_val.squeeze(1)
                    output_val = output_val.view(-1)                                                 
                    val_labels=val_labels.float()
                    val_pred = torch.ge(torch.sigmoid(output_val), torch.tensor(0.5)).float()
                    if config_params['classimbalance']=='focalloss':
                        loss1 = sigmoid_focal_loss(output_val, val_labels, alpha=-1, reduction='mean').item()
                    else:
                        loss1 = lossfn1(output_val, val_labels).item()
                elif config_params['activation'] == 'softmax':
                    val_pred = output_val.argmax(dim=1, keepdim=True)
                    loss1 = lossfn1(output_val, val_labels).item()
            
            if batch_val_no==0:
                val_pred_all = val_pred
                val_labels_all = val_labels
                print(output_val.data.shape, flush=True)
                if config_params['activation'] == 'sigmoid':
                    output_all_ten = torch.sigmoid(output_val.data)
                elif config_params['activation'] == 'softmax':
                    output_all_ten = F.softmax(output_val.data,dim=1)
                    if config_params['numclasses'] < 3:
                        output_all_ten = output_all_ten[:,1]
            else:
                val_pred_all = torch.cat((val_pred_all,val_pred),dim=0)
                val_labels_all = torch.cat((val_labels_all,val_labels),dim=0)
                if config_params['activation'] == 'sigmoid':
                    output_all_ten = torch.cat((output_all_ten,torch.sigmoid(output_val.data)),dim=0)
                elif config_params['activation'] == 'softmax':
                    if config_params['numclasses'] < 3:
                        output_all_ten = torch.cat((output_all_ten,F.softmax(output_val.data,dim=1)[:,1]),dim=0)
                    else:
                        output_all_ten = torch.cat((output_all_ten,F.softmax(output_val.data,dim=1)),dim=0)

            s = s+val_labels.shape[0]    
            val_loss += val_labels.size()[0]*loss1 # sum up batch loss
            correct, total_images, conf_mat_val, _ = evaluation.conf_mat_create(val_pred, val_labels, correct, total_images, conf_mat_val, config_params['classes'])
            
            batch_val_no+=1
            print('Val: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config_params['maxepochs'], batch_val_no, batches_val, loss1), flush=True)
    
    print("conf_mat_val:",conf_mat_val, flush=True)
    print("total_images:",total_images, flush=True)
    print("s:",s,flush=True)
    print('\nVal set: total val loss: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Epoch:{}\n'.format(
        val_loss, val_loss/total_images, correct, total_images,
        100. * correct / total_images,epoch+1), flush=True)
    
    auc = metrics.roc_auc_score(val_labels_all.cpu().numpy(), output_all_ten.cpu().numpy(), multi_class='ovo')
    return correct, total_images, val_loss, conf_mat_val, auc

if __name__=='__main__':
    #read arguments
    parser = argparse.ArgumentParser(description='CBIS-DDSM Training')
    parser.add_argument(
        "--config_file_path",
        type=str,
        default='/homes/bahadir.eryilmaz/repos/Masterarbeit/master_thesis_code/mammography/task_3/experiment_setups/direct_ssh',
        help="full path where the config.ini file containing the parameters to run this code is stored",
    )

    parser.add_argument(
        "--num_config_start",
        type=int,
        default=0,
        help="file number of hyperparameter combination to start with; one config file corresponds to one hyperparameter combination",
    )

    parser.add_argument(
        "--num_config_end",
        type=int,
        default=1,
        help="file number of hyperparameter combination to end with; one config file corresponds to one hyperparameter combination",
    )

    parser.add_argument('--config', default='/homes/bahadir.eryilmaz/repos/Masterarbeit/master_thesis_code/mammography/task_3/experiment_setups/direct_ssh/cfg_60.yaml', type=str, help='config file')
    parser.add_argument('--device', default=3, type=int, help='cuda device number')
    parser.add_argument('--cuda',default=None, type=int, help='cuda visible device set by slurm')
    parser.add_argument('--username',default=None, type=str, help='Username')
    parser.add_argument('--job_id',default=None, type=str, help='Job ID')
    parser.add_argument('--job_name',default=None, type=str, help='Job name')
    parser.add_argument('--node_list',default=None, type=str, help='Node list')
    parser.add_argument('--total_tasks', default=None,type=str, help='Total tasks')
    parser.add_argument('--submit_host',default=None, type=str, help='Submit host')
    parser.add_argument('--current_date', default=None,type=str, help='Current date')
    parser.add_argument('--working_directory',default=None, type=str, help='Working directory')
    args = parser.parse_args()

    num_config_start = args.num_config_start
    num_config_end = args.num_config_end

    # Access the parsed arguments
    config_path = args.config
    cuda_visible_devices = args.cuda
    username = args.username
    job_id = args.job_id
    job_name = args.job_name
    node_list = args.node_list
    total_tasks = args.total_tasks
    submit_host = args.submit_host
    current_date = args.current_date
    working_directory = args.working_directory

    if cuda_visible_devices is None: # Run from ssh access
        # CUDA device selection 
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    config_filename = os.path.splitext(os.path.basename(config_path))[0]
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader) 

    # WandB – Initialize a new run

    CONFIG = {'Task Name': cfg['TASK'], 
          '_wandb_kernel': 'aot'}

    #read all instructed config files
    config_file_names = glob.glob(args.config_file_path+'/config*')
    config_file_names = sorted(config_file_names, key=lambda x: int(re.search(r'\d+$', x.split('.')[-2]).group()))
    print("config files to be read:",config_file_names[num_config_start:num_config_end])
    
    config_file = config_file_names[0]

    # Modified from the reference code! We use only one config file at a time
    config_params = read_config_file.read_config_file(config_file)
    config_params['path_to_output'] = "/".join(config_file.split('/')[:-1])
    #g = set_random_seed(config_params) # Deactivate this

    # Set the random seed
    seeder = utils.Seeder(seed=cfg['seed'], backends=cfg['backends'])
    seeder.seed_everything()

    g = torch.Generator()
    g.manual_seed(cfg['seed'])

    # copy the infos in config_params to cfg
    for key in config_params.keys():
        cfg[key] = config_params[key]

    print("cfg:",cfg, flush=True)

    # Preprocessing and preparing the setup
    path_to_model, path_to_results_xlsx, path_to_results_text, path_to_learning_curve, path_to_log_file, path_to_hyperparam_search = output_files_setup.output_files(config_file, config_params, num_config_start, num_config_end)
    df_train, df_val, df_test, batches_train, batches_val, batches_test, view_group_indices_train = read_input_file.input_file_creation(config_params)
    dataloader_train, dataloader_val, dataloader_test = data_loader.dataloader(config_params, df_train, df_val, df_test, view_group_indices_train, g, seeder)
    
    begin_time = datetime.datetime.now()
    model, total_params = model_initialization(config_params)

    # 🐝 W&B Tracking
    RUN_CONFIG = CONFIG.copy()
    params = dict(run_name=cfg['RUN_NAME'],
                  model=cfg['femodel'], 
                  version=cfg['VERSION'],
                  config_filename=config_filename,
                  seed=torch.initial_seed(),
                  backend=cfg['backends'],
                  num_workers=cfg['numworkers'],
                  num_classes=cfg['numclasses'],
                  optimizer=cfg['optimizer_fn'],
                  epochs=cfg['numepochs'], 
                  batch=cfg['batchsize'],
                  lr=cfg['lr'],
                  weight_decay=cfg['wtdecay'],
                  groundtruthdic = cfg['groundtruthdic'],
                  resize=cfg['resize'],
                  datasplit=cfg['datasplit'],
                  activation=cfg['activation'],
                  pretrained=cfg['pretrained'],
                  dataset=cfg['dataset'],
                  trainingmethod = cfg['trainingmethod'])
    RUN_CONFIG.update(params)
    run = wandb.init(project='Task3_v3', config=RUN_CONFIG)
    

    device = logging_lib.start_log(f'{config_filename}',
                        cfg['OUTPUT_BASE'],
                        cfg['RUN_NAME'],
                        cfg['DEVICE'],
                        args)
    
    #training the model
    train(config_params, model, path_to_model, dataloader_train, dataloader_val, batches_train, batches_val, df_train, cfg, config_filename)
 
    #test the model
    per_model_metrics = test.run_test(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_results_xlsx, 'test_results')
    
    # Create a dictionary to store the metrics
    metrics_dict = {}
    metrics_names = ['Loss', 'PrecisionBin', 'PrecisionMicro', 'PrecisionMacro', 'RecallBin', 'RecallMicro', 'RecallMacro', 'F1Bin', 'F1Micro', 'F1macro', 'F1wtmacro', 'Acc', 'Cohens Kappa', 'AUC']

    # Assign the metrics values to their respective names
    for i, name in enumerate(metrics_names):
        metrics_dict[name] = per_model_metrics[i]

    # Log the metrics dictionary that we have it at the end of each epoch
    wandb.log(metrics_dict)


    f = open(path_to_log_file,'a')
    f.write("Model parameters:"+str(total_params/math.pow(10,6))+'\n')
    f.write("Start time:"+str(begin_time)+'\n')
    f.write("End time:"+str(datetime.datetime.now())+'\n')
    f.write("Execution time:"+str(datetime.datetime.now() - begin_time)+'\n')
    f.close()