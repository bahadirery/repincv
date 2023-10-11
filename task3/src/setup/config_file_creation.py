# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:06:36 2022

@author: PathakS
"""

from configparser import ConfigParser
import os
#import hyperparameter_optim

#GMIC reproduction hyperparameters
#hyperparam_config = [{'lr': 3.1622776601683795e-05, 'wtdecay':	0.0001, 'sm_reg_param':	0.000125893},{'lr':6.30957E-05, 'wtdecay': 0.000316228, 'sm_reg_param':	0.000158489},{'lr':7.94328E-05, 'wtdecay': 7.94328E-05, 'sm_reg_param': 0.000158489},{'lr':7.94328E-05,'wtdecay':3.98107E-05,'sm_reg_param':5.01187E-06}, {'lr': 0.0001, 'wtdecay': 3.16228E-05, 'sm_reg_param': 7.94328E-05}]
#GMIC final hyperparameter
hyperparam_config=[{'lr':3.1622776601683795e-05, 'wtdecay':3.9810717055349695e-05, 'sm_reg_param': False}]
#hyperparam_config=[{'lr':0.01, 'wtdecay':0.0005, 'sm_reg_param': False}]
#ResNet34 reproduction hyperparameter
#hyperparam_config = [{'lr': 1.584893192461114e-05, 'wtdecay': 0.0001},{'lr':1.584893192461114e-05, 'wtdecay': 1e-05},{'lr': 1.9952623149688786e-05, 'wtdecay': 5.011872336272725e-05},{'lr':1.9952623149688786e-05,'wtdecay':1.9952623149688786e-05}, {'lr': 3.1622776601683795e-05, 'wtdecay': 3.9810717055349695e-05}]
#hyperparam_config = [{'lr': 3.1622776601683795e-05, 'wtdecay': 3.9810717055349695e-05}]

names=[]
start = 0
end =  len(hyperparam_config)
count = 0

for hyperparam in hyperparam_config[start:end]:
    #Get the configparser object
    config_object = ConfigParser()
    #Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
    config_object["parametersetting"] = {
            "modelid": 31,
            "run": 4,
            "attention": False,  #options = imagewise, breastwise, False
            "dependency": False,
            "selfatt-nonlinear": False,
            "selfatt-gamma": False,
            "milpooling": False, #options=maxpool, average, attention, gatedattention, concat/ ismax, ismean, isatt, isgatt, esmax, esmean, esatt, esgatt
            "activation": 'sigmoid', #options = sigmoid, softmax
            "viewsinclusion": 'all', #option = standard, all -> change this to viewsinclusion: standard, all; in SIL: standard means all views. I put standard to prevent the dynamic training part of the code.
            "classimbalance": 'poswt', #options = wtcostfunc, poswt, oversampling, focalloss,False
            "optimizer": 'Adam', #options = SGD, Adam
            "patienceepochs": False, #10
            "usevalidation": True,
            "batchsize": 10, #options=10, 20
            "numclasses": 1,
            "maxepochs": 40, #150
            "numworkers": 8,
            "lr": 0.00075, #10**float(hyperparam['lr']), #0.001, 0.00002
            "wtdecay": float(hyperparam['wtdecay']), #10**float(hyperparam['wtdecay']), #0.0005, 0.00001
            "sm_reg_param": False, #10**float(hyperparam['sm_reg_param']), False
            "groundtruthdic": {'benign': 0, 'malignant': 1}, #{'normal':0,'benign':1,'malignant':2},
            "classes": [0, 1], #[0,1,2],
            "resize": [2944,1920], #options=1600, zgt, cbis-ddsm: [2944,1920], vindr:[2700, 990], None (for padding to max image size )
            "cam_size": False, #vindr: (85, 31), zgt:(92, 60)
            "crop_shape": False, #(256, 256)
            "dataaug": 'gmic', #options=small, big, wang, gmic, kim, shu
            "imagecleaning": 'own',
            "datasplit": 'officialtestset', #options: officialtestset, casebasedtestset
            "datascaling": 'scaling', #options=scaling, standardize, standardizeperimage,False
            "flipimage": True,
            "randseedother": 8, #options=8, 24, 80
            "randseeddata": 8, #options=8, 24, 80, 42
            "device": 'cuda',
            "trainingmethod": 'cosineannealing', #options: multisteplr1, fixedlr, lrdecayshu, lrdecaykim, cosineannealing
            "channel": 3, #options: 3 for rgb, 1 for grayscale
            "regionpooling": 'avgpool', #options: shu_ggp, shu_rgp, avgpool, maxpool, 1x1conv, t-pool
            "femodel": 'resnet34', #options: resnet50pretrainedrgbwang, densenet169, gmic_resnet18
            "pretrained": True, #options: True, False
            "topkpatch": False, #options: 0.02, 0.03, 0.05, 0.1
            "ROIpatches": False, #options: any number, 6 from gmic paper
            "learningtype": 'SIL', #options = SIL, MIL, MV (multiview)
            "dataset": 'cbis-ddsm', #options = cbis-ddsm, zgt, vindr
            "bitdepth": 16, #options: 8, 16
            "labeltouse": 'imagelabel', #options: imagelabel, caselabel
            "SIL_csvfilepath": "/projects/dso_mammovit/project_kushal/data/cbis-ddsm_singleinstance_groundtruth.csv", #"/groups/dso/spathak/vindr/MG_training_files_vindr_singleinstance_groundtruth.csv", #, #", #"/projects/dso_mammovit/project_kushal/data/cbis-ddsm_singleinstance_groundtruth.csv", #, #, #, #, #,
            "MIL_csvfilepath": "/projects/dso_mammovit/project_kushal/data/cbis-ddsm_multiinstance_groundtruth.csv", #"/groups/dso/spathak/vindr/MG_training_files_vindr_multiinstance_groundtruth.csv", #, #, #"/projects/dso_mammovit/project_kushal/data/cbis-ddsm_multiinstance_groundtruth.csv", #, #, #"/homes/spathak/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_viewnames_biradscombined_breastdensityadded_patientinfoadded_final4.csv", #
            "preprocessed_imagepath": "/projects/dso_mammovit/project_kushal/data/multiinstance_data_16bit", #"/groups/dso/spathak/vindr/processed_png_16bit", #, #, #"/projects/dso_mammovit/project_kushal/data/multiinstance_data_16bit", #, #, #, #, #, #"/projects/dso_mammovit/project_kushal/data/multiinstance_data_8bit", #, #"/groups/dso/spathak/vindr/processed_png_8bit", #, #, #"/local/work/spathak/zgt",
            "valloss_resumetrain": False,
            "papertoreproduce": False,
            "early_stopping_criteria": 'loss',
            "extra": False #options: dynamic_training
    }
    count+=1
    filename=''

    for key in config_object["parametersetting"].keys():
        print(key, config_object["parametersetting"][key])
        if key in ['modelid', 'attention', 'dependency', 'milpooling', 'femodel', 'viewsinclusion', 'papertoreproduce', 'learningtype', 'extra']:# 'regionpooling',]:
            #print(key, config_object["parametersetting"][key])
            if config_object["parametersetting"][key]!='False':
                if filename=='':
                    filename=key+config_object["parametersetting"][key]
                else:
                    filename=filename+'_'+key+config_object["parametersetting"][key]
    
    print(filename)

    config_object["parametersetting"]['filename']=filename
    if config_object["parametersetting"]['dataset'] == 'cbis-ddsm':
        path_to_output="/homes/bahadir.eryilmaz/repos/Masterarbeit/master_thesis_code/mammography/task_3/experiment_setup"+filename+"/"

    #create output_folder path
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    if str(config_object["parametersetting"]["randseeddata"])!=str(config_object["parametersetting"]["randseedother"]):
        rand_seed = str(config_object["parametersetting"]["randseedother"]) +'_'+ str(config_object["parametersetting"]["randseeddata"])
    else:
        rand_seed = str(config_object["parametersetting"]["randseeddata"])

    #Write the above sections to config.ini file
    if str(config_object["parametersetting"]["run"])!='False':
        with open(path_to_output+'config_'+rand_seed+'_'+'run_'+str(config_object["parametersetting"]["run"])+'.ini', 'w') as conf:
            config_object.write(conf)
    else:
        with open(path_to_output+'config_'+rand_seed+'.ini', 'w') as conf:
            config_object.write(conf)