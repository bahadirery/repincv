#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:20:38 2021

@author: spathak
"""

from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import os
import pickle
import torch
import math
import copy
#from skimage import io
from PIL import Image
from PIL import ImageOps
import numpy as np
import sys
import glob
import random
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn import metrics, utils
from collections import Counter
import operator
from sklearn.model_selection import GroupShuffleSplit
from torch.autograd import Variable
from copy import deepcopy
import imageio
import re

import logging
import pkg_resources
import subprocess
import tabulate
import platform
import cpuinfo
import io
import contextlib
from torchsummary import summary

#from data_loading import augmentations, loading

views_allowed_gmic = ['L-CC','L-MLO','R-CC','R-MLO']
cbis_view_dic = {'LEFT_CC': 'LCC', 'RIGHT_CC': 'RCC', 'LEFT_MLO': 'LMLO', 'RIGHT_MLO': 'RMLO'}

class MyCrop:
    """Randomly crop the sides."""

    def __init__(self, left=100,right=100,top=100,bottom=100):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __call__(self, x):
        width, height=x.size
        size_left = random.randint(0,self.left)
        size_right = random.randint(width-self.right,width)
        size_top = random.randint(0,self.top)
        size_bottom = random.randint(height-self.bottom,height)
        x = TF.crop(x,size_top,size_left,size_bottom,size_right)
        return x
    
class MyGammaCorrection:
    def __init__(self, factor=0.2):
        self.lb = 1-factor
        self.ub = 1+factor

    def __call__(self, x):
        gamma = random.uniform(self.lb,self.ub)
        return TF.adjust_gamma(x,gamma)

def myhorizontalflip(image,breast_side):
    if breast_side=='R':
        image = np.fliplr(image).copy() #R->L (following GMIC)
    return image

class MyHorizontalFlip:
    """Flip horizontally."""

    def __init__(self):
        pass

    def __call__(self, x, breast_side):
        #if breast_side=='L':
        #    return TF.hflip(x) #L->R
        if breast_side=='R':
            return TF.hflip(x) #R->L (following GMIC)
        else:
            return x

class MyPadding:
    def __init__(self, breast_side, max_height, max_width, height, width):
        self.breast_side = breast_side
        self.max_height=max_height
        self.max_width=max_width
        self.height=height
        self.width=width
          
    def __call__(self,img):
        print(img.shape)
        print(self.max_height-self.height)
        if self.breast_side=='L':
            image_padded=F.pad(img,(0,self.max_width-self.width,0,self.max_height-self.height,0,0),'constant',0)
        elif self.breast_side=='R':
            image_padded=F.pad(img,(self.max_width-self.width,0,0,self.max_height-self.height,0,0),'constant',0)
        print(image_padded.shape)
        return image_padded

class MyPaddingLongerSide:
    def __init__(self, resize):
        self.max_height=resize[0]
        self.max_width=resize[1]
        
    def __call__(self, img, breast_side):
        width=img.size[0]
        height=img.size[1]
        if height<self.max_height:
            diff=self.max_height-height
            img=TF.pad(img,(0,math.floor(diff/2),0,math.ceil(diff/2)),0,'constant')
        if width<self.max_width:
            diff=self.max_width-width
            if breast_side == 'L':
                img=TF.pad(img,(0,0,diff,0),0,'constant')
            elif breast_side == 'R':
                img=TF.pad(img,(diff,0,0,0),0,'constant')
        return img
        
class BreastCancerDataset_generator(Dataset):
    def __init__(self, config_params, df, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform
        self.config_params = config_params

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if self.config_params['learningtype'] == 'SIL':
            data = self.df.iloc[idx]
            img = collect_images(self.config_params, data)
            if self.transform:
                img=self.transform(img)
            
            if self.config_params['channel'] == 1:
                img=img[0,:,:]
                img=img.unsqueeze(0).unsqueeze(1)
            elif self.config_params['channel'] == 3:
                img=img.unsqueeze(0)
            return idx, img, torch.tensor(self.config_params['groundtruthdic'][data['Groundtruth']]), data['Views']
        
class CustomWeightedRandomSampler(Sampler):
    def __init__(self, config_params, df_train):
        self.df = df_train
        self.config_params = config_params
    
    def __iter__(self):
        len_instances=self.df.shape[0]
        OriginalIndices=np.array(range(0,len_instances))
        class_count=stratified_class_count(self.df)
        class_count=class_count.to_dict()
        class_count[0] = class_count.pop('benign')
        class_count[1] = class_count.pop('malignant')
        max_class = max(class_count.items(), key=operator.itemgetter(1))
        diff_count_class=max_class[1]-np.array([class_count[0],class_count[1]])
        labels=self.df['Groundtruth'].replace(self.config_params['groundtruthdic'])
        #print(np.append(OriginalIndices,np.random.choice(np.where(labels==0)[0],5),axis=0))
        for i in range(0,2):
            if i!=max_class[0]:
                repeater=int(diff_count_class[i]/class_count[i])
                OriginalIndices=np.append(OriginalIndices,np.repeat(np.where(labels==i),repeater))
                leftover=diff_count_class[i]-(class_count[i]*repeater)
                OriginalIndices=np.append(OriginalIndices,np.random.choice(np.where(labels==i)[0],leftover),axis=0)
                #OriginalIndices=np.append(OriginalIndices,np.random.choice(np.where(labels==i)[0],diff_count_class[i]),axis=0)
        random.shuffle(OriginalIndices)
        iter_shuffledIndex=iter(OriginalIndices)
        return iter_shuffledIndex
    
    def __len__(self):
        class_count=stratified_class_count(self.df)
        class_count=class_count.to_dict()
        max_class=max(class_count.items(), key=operator.itemgetter(1))
        len_instances_oversample=max_class[1]*2
        #print("Weighted Sampler __len__:",len_instances_oversample)
        return len_instances_oversample

def MyCollate(batch):
    i=0
    index=[]
    target=[]
    for item in batch:
        if i==0:
            data = batch[i][1]
            views_names = [item[3]]
        else:
            data=torch.cat((data,batch[i][1]),dim=0)
            views_names.append(item[3])
        index.append(item[0])
        target.append(item[2])
        i+=1
    index = torch.LongTensor(index)
    target = torch.LongTensor(target)
    return [index, data, target, views_names]

##--------------------------------------------------------------- collect image ----------------------------------------------------#
def view_extraction(series_list, views_allowed):
    series_list = list(map(lambda x: [x.split('_')[0].replace(' ', '').upper(), x], series_list))
    series_list = list(filter(lambda x: x[0] in views_allowed, series_list))
    series_list = sorted(series_list, key=lambda x: x[0])
    return series_list

def selecting_data(config_params, img_list):
    if config_params['dataset'] == 'zgt':
        if config_params['bitdepth'] == 8:
            img_list = list(filter(lambda x: re.search('_processed.png$', x), img_list))
        elif config_params['bitdepth'] == 12:
            img_list = list(filter(lambda x: re.search('_processed.npy$', x), img_list))
    return img_list

def views_allowed_dataset(config_params):
    if config_params['dataset'] == 'zgt' and config_params['viewsinclusion'] == 'all':
        views_allowed=['LCC', 'LLM', 'LML', 'LMLO', 'LXCCL', 'RCC', 'RLM', 'RML', 'RMLO', 'RXCCL']
    else:
        views_allowed = ['LCC','LMLO','RCC','RMLO']
    return views_allowed

def collect_cases(config_params, data):
    views_allowed = views_allowed_dataset(config_params)
    breast_side=[]
    image_read_list=[]
    views_saved=[]
    data1 = {}
    studyuid_path = str(data['FullPath'])
    series_list = os.listdir(studyuid_path)
    series_list = view_extraction(series_list, views_allowed)
    #series_list.sort()
    if series_list[0][1].split('.')[-1] == 'png':
        for series in series_list:
            img_path = studyuid_path + '/' + series[1]
            data1['FullPath'] = img_path
            data1['Views'] = series[0]
            img = collect_images(config_params, data1)
            if series[0] in views_allowed and series[0] not in views_saved:
                views_saved.append(series[0])
                image_read_list.append(img)
                breast_side.append(series[0][0])
    else:
        for series in series_list:
            series_path = studyuid_path+'/'+series[1]
            img_list = os.listdir(series_path)
            img_list = selecting_data(config_params, img_list)
            for image in img_list:
                img_path = series_path+'/'+image
                data1['FullPath'] = img_path
                data1['Views'] = series[0]
                img = collect_images(config_params, data1)
                if series[0] in views_allowed and series[0] not in views_saved:
                    if config_params['dataset']  == 'zgt' and config_params['viewsinclusion']=='all' and config_params['bitdepth']==12: #solve this in future
                        if series[0] in data['Views'].split('+'):
                            views_saved.append(series[0])
                            image_read_list.append(img)
                            breast_side.append(series[0][0])
                    else:
                        views_saved.append(series[0])
                        image_read_list.append(img)
                        breast_side.append(series[0][0])
        
        #if "+".join(sorted(data['Views'].split('+')))!="+".join(views_saved):
        #    print("Not matched:", data['FullPath'], flush=True)
        #    print(data['Views'], flush=True)
        #    print(views_saved, flush=True)
    return image_read_list, breast_side, views_saved

def collect_images(config_params, data):
    views_allowed = views_allowed_dataset(config_params)
    if config_params['bitdepth'] ==  8:
        img = collect_images_8bits(config_params, data, views_allowed)
    elif config_params['bitdepth'] == 16:
        if config_params['imagecleaning'] == 'gmic':
            img = collect_images_gmiccleaningcode(data)
        else:
            img = collect_images_16bits(config_params, data, views_allowed)
    elif config_params['bitdepth'] == 12:
        img = collect_images_12bits(config_params, data, views_allowed)
    return img
         
def collect_images_8bits(config_params, data, views_allowed):
    #collect images for the model
    if data['Views'] in views_allowed:
        img_path = str(data['FullPath'])
        img = cv2.imread(img_path, 0)
        img_dtype = img.dtype
        if config_params['dataset']=='cbis-ddsm':
            clahe_create = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe_create.apply(img)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).astype(np.float32)
        breast_side = data['Views'][0]
        if img_dtype=='uint8':
            img/=255
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        if config_params['flipimage']:
            hflip_img = MyHorizontalFlip()
            img = hflip_img(img, breast_side)
        return img
    else:
        print('error in view')
        sys.exit()

def collect_images_16bits(config_params, data, views_allowed):
    #collect images for the model
    if data['Views'] in views_allowed:
        img_path = str(data['FullPath'])
        #print('img path:', img_path)
        img = cv2.imread(img_path,-1)
        img_dtype = img.dtype
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).astype(np.float32)
        breast_side = data['Views'][0]
        if img_dtype=='uint16':
            img/=65535
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        if config_params['flipimage']:
            hflip_img = MyHorizontalFlip()
            img = hflip_img(img,breast_side)
        return img
    else:
        print('error in view')
        sys.exit()

def collect_images_12bits(config_params, data, views_allowed):
    #collect images for the model
    #print(data['Views'])
    #print(views_allowed)
    if data['Views'] in views_allowed:
        img_path = str(data['FullPath'])
        img = np.load(img_path).astype(np.float32)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        breast_side = data['Views'][0]
        img/=4095
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        if config_params['flipimage']:
            hflip_img = MyHorizontalFlip()
            img = hflip_img(img,breast_side)
        return img
    else:
        print(data['FullPath'])
        print('error in view')
        sys.exit()
        
##-------------------------------------------------------------------- collect images end ----------------------------------------------------------------------#

##------------------------------------------------------------------ different types of data augmentation functions --------------------------------------------# 
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def data_augmentation_train_shen_gmic(config_params, mean, std_dev):
    preprocess_train_list=[]

    if config_params['imagecleaning']!='gmic':
        if config_params['resize']:
            preprocess_train_list.append(transforms.Resize((config_params['resize'][0], config_params['resize'][1]), antialias=None))
    
    if config_params['papertoreproduce'] == 'wu20':
        noise_std = 0.01
    else:
        noise_std = 0.005

    preprocess_train_list=preprocess_train_list+[
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=(15),translate=(0.1,0.1),scale=(0.8,1.6),shear=(25)),
        AddGaussianNoise(mean=0, std=noise_std)
        ]
    
    if config_params['datascaling']:
        if config_params['datascaling']!='standardizeperimage':
            preprocess_train_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_train = transforms.Compose(preprocess_train_list)
    return preprocess_train

def data_augmentation_test_shen_gmic(config_params, mean, std_dev):
    preprocess_test_list=[]

    if config_params['imagecleaning']!='gmic':
        if config_params['resize']:
            preprocess_test_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1]), antialias=None))
    
    if config_params['datascaling']:  
        if config_params['datascaling']!='standardizeperimage':
            preprocess_test_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_test = transforms.Compose(preprocess_test_list)
    return preprocess_test

def data_augmentation(config_params):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            utils.MyCrop(),
            transforms.Pad(100),
            transforms.RandomRotation(3),
            transforms.ColorJitter(brightness=0.20, contrast=0.20),
            transforms.RandomAdjustSharpness(sharpness_factor=0.20),
            utils.MyGammaCorrection(0.20),
            utils.MyPaddingLongerSide(config_params['resize']),
            transforms.Resize((config_params['resize'][0],config_params['resize'][1]), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #'train': transforms.Compose([
        #    transforms.Resize((resize,resize)),
        #    transforms.ToTensor(),
        #    transforms.RandomHorizontalFlip(p=0.5),
        #    transforms.ColorJitter(contrast=0.20, saturation=0.20),
        #    transforms.RandomRotation(30),
        #    AddGaussianNoise(mean=0, std=0.005),
        #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #]),
        'val': transforms.Compose([
            transforms.Resize((config_params['resize'][0],config_params['resize'][1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

def data_augmentation_train(config_params, mean, std_dev):
    preprocess_train_list=[]

    preprocess_train_list=preprocess_train_list+[
        MyCrop(),
        #transforms.Pad(100),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.RandomAdjustSharpness(sharpness_factor=0.20),
        MyGammaCorrection(0.20),
        MyPaddingLongerSide(config_params['resize'])]
    
    if config_params['resize']:
        preprocess_train_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1]), antialias=None))
    
    preprocess_train_list.append(transforms.ToTensor())
    
    if config_params['datascaling']:
        if config_params['datascaling']!='standardizeperimage':
            preprocess_train_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_train = transforms.Compose(preprocess_train_list)
    return preprocess_train

def data_augmentation_test(config_params, mean, std_dev):
    preprocess_test_list=[]

    if config_params['resize']:
        preprocess_test_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1]), antialias=None))
    
    preprocess_test_list.append(transforms.ToTensor())
    
    if config_params['datascaling']:  
        if config_params['datascaling']!='standardizeperimage':
            preprocess_test_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_test = transforms.Compose(preprocess_test_list)
    return preprocess_test

##------------------------------------------------------------------ different types of data augmentation functions end --------------------------------------------#

def plot(filename):
    df=pd.read_excel(filename).sort_values(by=['Count'],ascending=False)
    print(df['Views'].tolist())
    print(df['Count'].tolist())
    plt.figure(figsize=(5,5))
    plt.bar(df['Views'].tolist(),df['Count'].tolist())
    plt.xticks(rotation=45,ha='right')
    plt.savefig('view_distribution.png', bbox_inches='tight')    

def stratified_class_count(df):
    class_count=df.groupby(by=['Groundtruth']).size()
    return class_count

def class_distribution_weightedloss(config_params, df):
    df_groundtruth=df['Groundtruth'].map(config_params['groundtruthdic'])
    class_weight=utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.array(config_params['classes']), y = df_groundtruth)
    print("class count:", dict(Counter(df_groundtruth)))
    print("class weight:", class_weight)
    return torch.tensor(class_weight,dtype=torch.float32)

def class_distribution_poswt(df):
    class_count=df.groupby(by=['Groundtruth']).size()
    pos_wt=torch.tensor([float(class_count['benign'])/class_count['malignant']],dtype=torch.float32)
    print(pos_wt)
    return pos_wt

def stratifiedgroupsplit(df, rand_seed):
    groups = df.groupby('Groundtruth')
    all_train = []
    all_test = []
    all_val = []
    train_testsplit = GroupShuffleSplit(test_size=0.15, n_splits=2, random_state=rand_seed)
    train_valsplit = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=rand_seed)
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group['Patient_Id'].isin(all_train+all_val+all_test)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds1, test_inds = next(train_testsplit.split(group, groups=group['Patient_Id']))
        train_inds, val_inds = next(train_valsplit.split(group.iloc[train_inds1], groups=group.iloc[train_inds1]['Patient_Id']))
    
        all_train += group.iloc[train_inds1].iloc[train_inds]['Patient_Id'].tolist()
        all_val += group.iloc[train_inds1].iloc[val_inds]['Patient_Id'].tolist()
        all_test += group.iloc[test_inds]['Patient_Id'].tolist()
        
    train = df[df['Patient_Id'].isin(all_train)]
    val = df[df['Patient_Id'].isin(all_val)]
    test = df[df['Patient_Id'].isin(all_test)]
    
    return train, val, test

def calculate_image_size(df):
    total=df.shape[0]
    w_all=[]
    h_all=[]
    count_less=0
    count_more=0
    count_wless=0
    count_wmore=0
    
    for k in range(total):
        if k%5==0:
            print(str(k)+"/"+str(total))
        data=df.iloc[k]
        #studyuid_path = str(df.iloc[k]['FullPath'])
        #series_list = os.listdir(studyuid_path)
        img, _ =collect_images(data)
        w,h=img.size
        w_all.append(w)
        h_all.append(h)
        if w<1600 and h<1600:
            count_less+=1
        elif w>1600 and h>1600:
            count_more+=1
        elif w<1600 and h>1600:
            count_wless+=1
        elif w>1600 and h<1600:
            count_wmore+=1
    
    print("min w:", min(w_all))
    print("min h:", min(h_all))
    print("max w:", max(w_all))
    print("max h:", max(h_all))
    print("less than 1600,1600:",count_less)
    print("more than 1600,1600:",count_more)
    print("w less than 1600, h more than 1600:",count_wless)
    print("w more than 1600, h less than 1600:",count_wmore)
    w_mean_dataset = np.mean(np.array(w_all))
    w_std_dataset = np.std(np.array(w_all))
    h_mean_dataset = np.mean(np.array(h_all))
    h_std_dataset = np.std(np.array(h_all))
    return w_mean_dataset, w_std_dataset, h_mean_dataset, h_std_dataset

def calculate_dataset_mean_stddev(df, resize, transform):
    means = []
    stds = []
    total=df.shape[0]
    if transform:
        if resize:
            preprocess = transforms.Compose([
                transforms.Resize((resize[0],resize[1]), antialias=None),
                transforms.ToTensor()])
        else:
            preprocess = transforms.Compose([
                transforms.ToTensor()])
    
    for k in range(total):
        if k%5==0:
            print(str(k)+"/"+str(total))
        studyuid_path = str(df.iloc[k]['FullPath'])
        series_list = os.listdir(studyuid_path)
        image_list, _, _ = collect_images(studyuid_path,series_list)
        for j,img in enumerate(image_list):
            if transform:
                img=preprocess(img)
            means.append(torch.mean(img))
            stds.append(torch.std(img))
    
    mean_dataset = torch.mean(torch.tensor(means))
    std_dataset = torch.mean(torch.tensor(stds))
    return mean_dataset, std_dataset

def save_model(model, optimizer, epoch, loss, path_to_model):
    state = {
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, path_to_model)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def confusion_matrix_norm_func(conf_mat,fig_name,class_name):
    #class_name=['W','N1','N2','N3','REM']
    conf_mat_norm=np.empty((conf_mat.shape[0],conf_mat.shape[1]))
    #conf_mat=confusion_matrix(y_true, y_pred)
    for i in range(conf_mat.shape[0]):
        conf_mat_norm[i,:]=conf_mat[i,:]/sum(conf_mat[i,:])
    #print(conf_mat_norm)
    print_confusion_matrix(conf_mat_norm,class_name,fig_name)
    
def print_confusion_matrix(conf_mat_norm, class_names, fig_name, figsize = (2,2), fontsize=5):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    #sns.set()
    #grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    fig, ax = plt.subplots(figsize=figsize)
    #cbar_ax = fig.add_axes([.93, 0.1, 0.05, 0.77])
    #fig = plt.figure(figsize=figsize)
    heatmap=sns.heatmap(
        yticklabels=class_names,
        xticklabels=class_names,
        data=conf_mat_norm,
        ax=ax,
        cmap='YlGnBu',
        cbar=False,
        #cbar_ax=cbar_ax,
        annot=True,
        annot_kws={'size':fontsize},
        fmt=".2f",
        square=True
        #linewidths=0.75
        )
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel('True label',labelpad=0,fontsize=fontsize)
    ax.set_xlabel('Predicted label',labelpad=0,fontsize=fontsize)
    #cbar_ax.tick_params(labelsize=fontsize) 
    #ax.get_yaxis().set_visible(False)
    #plt.tight_layout()
    #plt.show()
    ax.set_title(fig_name)
    fig.savefig(fig_name+'.pdf', format='pdf', bbox_inches='tight')    


class Seeder():
    '''A class to handle seeding.'''
    def __init__(self, seed: int = 42, backends: bool = False):
        self.seed = seed
        self.backends = backends      

   
    def seed_everything(self):
        '''Seed everything for reproducibility'''

        # Set the random seed manually for reproducibility.
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)

        # Set the numpy seed manually for reproducibility.
        np.random.seed(self.seed)

        # Set the torch seed manually for reproducibility.
        torch.manual_seed(self.seed) # pytorch (both CPU and CUDA)
        if not self.backends:
            torch.backends.cudnn.benchmark = True

        # Limit the randomness in cuDNN backend.
        if self.backends:
            # https://pytorch.org/docs/stable/notes/randomness.html
            torch.cuda.manual_seed(self.seed) # pytorch (both CPU and CUDA)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            os.environ["CUDNN_DETERMINISTIC"] = "1"
            os.environ["CUDNN_BENCHMARK"] = "0"

    
    def seed_worker(self,worker_id):
        '''Seed a worker with the given ID. This function is called'''
        seed = self.seed % 2**32
        np.random.seed(seed)
        random.seed(seed)
        print(f"Worker {worker_id} seeded with {seed}")



