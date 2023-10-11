import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from time import time
import os
import yaml

from tqdm.auto import tqdm

from model import build_model
from dataset import get_datasets, get_data_loaders
from utils import *
from sklearn.metrics import precision_score, recall_score, f1_score
# WandB – Import the wandb library
import wandb
WANDB__SERVICE_WAIT=500

from test import *

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/task1/exp.yaml', type=str, help='config file')
parser.add_argument('--device', default=2, type=int, help='cuda device number')
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



# Training function.

def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    preds_list = []
    labels_list = []
    for i, data in enumerate(trainloader):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / len(trainloader)
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    # Calculate precision, recall and f1_score
    precision = precision_score(labels_list, preds_list, average='weighted')
    recall = recall_score(labels_list, preds_list, average='weighted')
    f1 = f1_score(labels_list, preds_list, average='weighted')
    return epoch_loss, epoch_acc, precision, recall, f1

def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            
            # Append batch prediction results
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    
    # Calculate precision, recall, f1_score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return epoch_loss, epoch_acc, precision, recall, f1

if __name__ == '__main__':


    config_filename = os.path.splitext(os.path.basename(config_path))[0]
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    CONFIG = {'Task Name': cfg['RUN_NAME'], 
              '_wandb_kernel': 'aot'}

    seeder = Seeder(seed=cfg['seed'], backends=cfg['backends'])
    seeder.seed_everything()

    g = torch.Generator()
    g.manual_seed(cfg['seed'])

    device = start_log(f'{config_filename}',
                        cfg['OUTPUT_BASE'],
                        cfg['RUN_NAME'],
                        cfg['DEVICE'],
                        args)
    best_acc = 0  # best test accuracy

    start_epoch = cfg['start_epoch']
    max_epochs = cfg['max_epochs']
    PATIENCE = cfg['patience']
    model_name = cfg['model_name']


    # Load the training and validation datasets.
    dataset_train, dataset_valid = get_datasets()
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(
        dataset_train=dataset_train, 
        dataset_valid=dataset_valid,
        batch_size=cfg['tr_batch_size'],
        num_workers=cfg['num_workers'],
    )

    # Learning_parameters.
    device = 'cuda'
    print(f"Computation device: {device}")
    print(f"Learning rate: {cfg['lr']}")
    print(f"Epochs to train for: {cfg['max_epochs']}\n")

    # Load the model.
    model = build_model(
        pretrained=False,
        fine_tune=True, 
        num_classes=cfg['num_classes'],
    ).to(device)

    # Print the model architecture
    log_model_info(model) 

    checkpoint_path = cfg['OUTPUT_BASE'] + '/weights/' + f'{config_filename}.pth'

    # Load checkpoint.
    if os.path.exists(checkpoint_path):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    if cfg['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'],
                              momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    if cfg['optimizer'] == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.851436, 0.999689), amsgrad=True)
    
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Initialize LR scheduler.
    lr_scheduler = LRScheduler(optimizer, patience=1, factor=0.5)


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
                  momentum=cfg['momentum'],
                  scheduler=cfg['scheduler'],)
    RUN_CONFIG.update(params)
    run = wandb.init(project='Task2_v2', config=RUN_CONFIG)

    wandb.watch(model, log_freq=100) # 🐝

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    best_valid_loss = float('inf')
    # Start the training.
    start_time = time()
    for epoch in range(cfg['start_epoch'], cfg['max_epochs']):
        print(f"[INFO]: Epoch {epoch+1} of {cfg['max_epochs']}")

        # Log GPU info
        log_gpu_info(epoch=epoch)

        train_epoch_loss, train_epoch_acc, tr_precision, tr_recall, tr_f1 = train(
            model, 
            train_loader, 
            optimizer, 
            criterion
        )

        # Log the end of the epoch metrics
        log_end_epoch(epoch, train_epoch_loss, train_epoch_acc, tr_precision, tr_recall, tr_f1)


        valid_epoch_loss, valid_epoch_acc, val_precision, val_recall, val_f1 = validate(
            model, 
            valid_loader, 
            criterion
        )

        state_dict = {
            'epoch': epoch + 1,
            'acc': valid_epoch_acc,
            'net': model.state_dict(),
            }
        
        checkpoint_dir = os.path.join(cfg['OUTPUT_BASE'], 'weights', config_filename)
        # Save the best model.
        if valid_epoch_loss < best_valid_loss:
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_valid_loss = valid_epoch_loss
            checkpoint_path_best = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path_best)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        lr_scheduler(valid_epoch_loss)
        print('-'*50)

        # Get the current lr
        current_lr = optimizer.param_groups[0]['lr']
        

        # Log metrics to wandb
        wandb.log({
            "Epoch": epoch,
            "Learning Rate": current_lr,
            "Train Loss": train_epoch_loss,
            "Train Acc": train_epoch_acc,
            "Valid Loss": valid_epoch_loss,
            "Valid Acc": valid_epoch_acc,
             "Precision": val_precision,
            "Recall": val_recall,
            "F1 Score": val_f1
        }, step = epoch)        

        # Save the model on every third epoch for weight analysis.
        if epoch % 3 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(state_dict, checkpoint_path)
            logging.info(f'Checkpoint saved to {checkpoint_path}')

    end_time = time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    

    # Training is completed now test the best model on the test set.
    print('TESTING BEST MODEL')

    # Test the model on the test set.
    df = pd.read_csv('/projects/datashare/dso/crack_detection/dataset/test.csv')
    X = df.image_path.values # Image paths.
    y = df.target.values # Targets
    dataset_test = ImageDataset(X, y, tfms=0)

    test_loader = DataLoader(
        dataset_test, batch_size=cfg['test_batch_size'], 
        shuffle=False, num_workers=cfg['num_workers']
    )
    # Load the model.
    model = build_model(
        pretrained=None,
        fine_tune=False, 
        num_classes=2
    ).to(DEVICE)
    # After training, load the best model parameters
    model.load_state_dict(torch.load(checkpoint_path_best))

    predictions_list, ground_truth_list, acc, precision, recall, f1 = test(
        model, test_loader, DEVICE
    )

    end_test = (acc, f1, precision, recall)

    log_results(end_test, elapsed_time)

    # Log metrics to wandb
    wandb.log({
        "Test Accuracy": acc,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1 Score": f1
    })  

    # save the config file to wandb
    log_file_path = os.path.join(cfg['OUTPUT_BASE'], 'logs', config_filename + '.log')

    # Save the model
    wandb.save(checkpoint_path_best, base_path=cfg['OUTPUT_BASE']) # 🐝

    # Save the log file
    wandb.save(log_file_path, base_path=cfg['OUTPUT_BASE']) # 🐝

    # Save the config file
    # wandb.save(config_path, base_path='/homes/bahadir.eryilmaz/repos/Masterarbeit/master_thesis_code/cifar_kaggle/configs') # 🐝

    # 🐝 Experiment End for the run
    wandb.finish()


    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')
