'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging
import yaml
import json
import pickle
import gzip

# WandB – Import the wandb library
import wandb
WANDB__SERVICE_WAIT=500

from preact_resnet import *
from utils import *
from time import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config', default='../configs/task1/exp.yaml', type=str, help='config file')

# Slurm arguments only applicable when running on the cluster
parser.add_argument('--device', default=None, type=int, help='cuda device number')
parser.add_argument('--cuda',default=0, type=int, help='cuda visible device set by slurm')
parser.add_argument('--username',default=None, type=str, help='Username')
parser.add_argument('--job_id',default=None, type=str, help='Job ID')
parser.add_argument('--job_name',default=None, type=str, help='Job name')
parser.add_argument('--node_list',default=None, type=str, help='Node list')
parser.add_argument('--total_tasks', default=None,type=str, help='Total tasks')
parser.add_argument('--submit_host',default=None, type=str, help='Submit host')
parser.add_argument('--current_date', default=None,type=str, help='Current date')
parser.add_argument('--working_directory',default=None, type=str, help='Working directory')

# Parse the command-line arguments
args = parser.parse_args()

if args.job_id is None: # Run from ssh access
    live_metrics = True
else:
    live_metrics = False

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

CONFIG = {'Task Name': cfg['RUN_NAME'], 
          '_wandb_kernel': 'aot'}


#To get reproducible experiments

#if cfg['seed'] != 'None':
#    seed_everything(cfg['seed'], backends=cfg['backends'])

#
seeder = Seeder(seed=cfg['seed'], backends=cfg['backends'])
seeder.seed_everything()

g = torch.Generator()
g.manual_seed(cfg['seed'])



device = log_basics(f'{config_filename}',
                    cfg['OUTPUT_BASE'],
                    cfg['RUN_NAME'],
                    cfg['DEVICE'],
                    args)

best_acc = 0  # best test accuracy

start_epoch = cfg['start_epoch']
max_epochs = cfg['max_epochs']
PATIENCE = cfg['patience']
model_name = cfg['model_name']

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the training and test datasets
trainset = torchvision.datasets.CIFAR10(
    root=cfg['INPUT_BASE'], train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=cfg['tr_batch_size'], shuffle=True, num_workers=cfg['num_workers'],
    worker_init_fn = seeder.seed_worker, generator = g)

testset = torchvision.datasets.CIFAR10(
    root=cfg['INPUT_BASE'], train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=cfg['test_batch_size'], shuffle=False, num_workers=cfg['num_workers'],
    worker_init_fn = seeder.seed_worker, generator = g)


# Move the model to the GPU
'''
Disabling the benchmarking feature with 'torch.backends.cudnn.benchmark = False'
causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.

However, if you do not need reproducibility across multiple executions of your application, 
then performance might improve if the benchmarking feature is enabled with 
'torch.backends.cudnn.benchmark = True'.
'''
# Model

if cfg['model_name'] == 'PreActResNet18':
    net = PreActResNet18()

if cfg['model_name'] == 'PreActResNet34':
    net = PreActResNet34()  

if cfg['model_name'] == 'PreActResNet50':
    net = PreActResNet50()


# Print the model architecture
log_model_info(net)    

# Load checkpoint.
if cfg['resume']:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint_path = cfg['OUTPUT_BASE'] + '/weights/' + f'{config_filename}.pth'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()

if cfg['optimizer'] == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=cfg['lr'],
                          momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
if cfg['optimizer'] == 'ADAM':
    optimizer = optim.Adam(net.parameters(), lr=cfg['lr'], betas=(0.851436, 0.999689), amsgrad=True,
                           )
if cfg['scheduler'] == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

if cfg['scheduler'] == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['max_epochs'])

# Move the model to the GPU
net = net.to(device)



# Training
def train(epoch):
    log_gpu_info(epoch=epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    layer_epoch_weights = {}  # Initialize an empty dictionary for storing layer weights
    
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if live_metrics:

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    
    # Calculate overall accuracy for the epoch
    accuracy = correct / total 
    
    # Log the accuracy for the epoch

    # Flatten the predicted and target tensors for calculating F1 score, precision, and recall
    predicted_flat = predicted.flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()


    # Calculate F1 score
    f1 = f1_score(targets_flat, predicted_flat, average='macro')

    # Calculate precision
    precision = precision_score(targets_flat, predicted_flat, average='macro',zero_division=1)

    # Calculate recall
    recall = recall_score(targets_flat, predicted_flat, average='macro')

    endloss = train_loss / (batch_idx + 1)
    

    return endloss, accuracy, f1, precision, recall, epoch

# Testing
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    all_predicted = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predicted.extend(predicted.tolist())
            all_targets.extend(targets.tolist())

            if live_metrics:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total


    # Flatten the predicted and target tensors for calculating F1 score, precision, and recall
    predicted_flat = predicted.flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()

    # Calculate F1 score
    f1 = f1_score(targets_flat, predicted_flat, average='macro')

    # Calculate precision
    precision = precision_score(targets_flat, predicted_flat, average='macro',zero_division=1)

    # Calculate recall
    recall = recall_score(targets_flat, predicted_flat, average='macro',zero_division=1)
    
    f1_2 = f1_score(all_targets, all_predicted, average='macro')
    recall_2 = recall_score(all_targets, all_predicted, average='macro')
    precision_2 = precision_score(all_targets, all_predicted, average='macro')
    

    # Log the test results
    log_string = f'\nAccuracy of the network on the 10000 test images: {acc} \n'
    log_string += f'F1 score of the network on the 10000 test images: {f1} \n'
    log_string += f'F1_2 score of the network on the 10000 test images: {f1_2} \n'
    log_string += f'Precision of the network on the 10000 test images: {precision} \n'
    log_string += f'Precision_2 of the network on the 10000 test images: {precision_2} \n'
    log_string += f'Recall of the network on the 10000 test images: {recall} \n'
    log_string += f'Recall_2 of the network on the 10000 test images: {recall_2} \n'
    logging.info(log_string)

    return acc, f1, precision, recall, epoch, f1_2, precision_2, recall_2



# 🐝 W&B Tracking
RUN_CONFIG = CONFIG.copy()
params = dict(run_name=cfg['RUN_NAME'],
              model=cfg['model_name'], 
              version=cfg['VERSION'],
              config_filename=config_filename,
              seed=torch.initial_seed(),
              backend=cfg['backends'],
              num_workers=cfg['num_workers'],
              num_classes=cfg['num_classes'],
              optimizer=cfg['optimizer'],
              epochs=cfg['max_epochs'], 
              batch=cfg['tr_batch_size'],
              lr=cfg['lr'],
              weight_decay=cfg['weight_decay'],
              momentum=cfg['momentum'],)
RUN_CONFIG.update(params)
run = wandb.init(project='Task1_v4', config=RUN_CONFIG)

wandb.watch(net, log_freq=100) # 🐝

best_accuracy = 0.0
best_f1_score = 0.0


start_time = time()
for epoch in range(start_epoch, start_epoch+max_epochs):

    # Train one epoch
    end_epoch = train(epoch)

    # Log the end of the epoch metrics
    log_end_epoch(end_epoch)
    
    
    # Test the model and log the results
    end_test = test(epoch)

    current_lr = optimizer.param_groups[0]['lr']

    # Update the learning rate
    scheduler.step()


    val_acc = end_test[0]
    val_f1 = end_test[5]

    # Check if current F1 score is better than the best F1 score
    if val_f1 > best_f1_score:
        best_f1_score = val_f1

    # Check if current accuracy is better than the best accuracy
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        
    # Save the checkpoint
    state = {
            'net': net.state_dict(),
            'acc': val_acc,
            'epoch': epoch,
        }

    if epoch % 5 == 0:
        checkpoint_dir = os.path.join(cfg['OUTPUT_BASE'], 'weights', config_filename)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save(state, checkpoint_path)
        logging.info(f'Checkpoint saved to {checkpoint_path}')


    wandb.log({"current_epoch": epoch}, step=epoch) # 🐝
    wandb.log({"train_loss": end_epoch[0]}, step=epoch) # 🐝
    wandb.log({"train_acc": end_epoch[1]},step=epoch) # 🐝
    wandb.log({"learning_rate ": current_lr}, step=epoch) # 🐝
    wandb.log({"valid_acc": val_acc}, step=epoch) # 🐝
    wandb.log({"valid_f1": end_test[1]}, step=epoch) # 🐝
    wandb.log({"valid_precision": end_test[2]}, step=epoch) # 🐝
    wandb.log({"valid_recall": end_test[3]}, step=epoch) # 🐝
    
    # Second F1, precision, and recall
    wandb.log({"valid_f1_2": end_test[5]}, step=epoch) # 🐝
    wandb.log({"valid_precision_2": end_test[6]}, step=epoch) # 🐝
    wandb.log({"valid_recall_2": end_test[7]}, step=epoch) # 🐝

    # Log the best metrics
    wandb.log({"best_accuracy": best_accuracy}, step=epoch) # 🐝
    wandb.log({"best_f1_score": best_f1_score}, step=epoch) # 🐝

end_time = time()
# Calculate the elapsed time
elapsed_time = end_time - start_time

format_time(elapsed_time)

# Log the end of the training
log_results(end_test, elapsed_time)

log_file_path = os.path.join(cfg['OUTPUT_BASE'], 'logs', config_filename + '.log')

training_weights_path = os.path.join(cfg['OUTPUT_BASE'], 'weights', config_filename + '_pkl.gz')

# Save the model
wandb.save(checkpoint_path, base_path=cfg['OUTPUT_BASE']) # 🐝

# Save the log file
wandb.save(log_file_path, base_path=cfg['OUTPUT_BASE']) # 🐝

# Save the config file
wandb.save(config_path, base_path='task1/configs') # 🐝

# 🐝 Experiment End for the run
wandb.finish()
