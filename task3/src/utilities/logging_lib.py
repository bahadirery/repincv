import os
import sys
import math

import torch.nn as nn
import torch.nn.init as init

import logging
import io
import sys
import wandb
import pkg_resources
import platform
import cpuinfo
import subprocess
import os
import torch
import numpy as np
from time import time, strftime, localtime
from tabulate import tabulate
import torch.nn as nn
from torchsummary import summary
import contextlib

def start_log(log_filename,output_base,run_name,device,args):
    ### Set up logging configuration ###
    log_file = os.path.join(output_base,'logs', log_filename + '.log')
    # Delete the old training.log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
    class PrintCapture:
        def __init__(self):
            self.content = ''
        def write(self, text):
            self.content += text
    # Create an instance of the PrintCapture class
    capture = PrintCapture()
    # Redirect the stdout to the PrintCapture instance
    sys.stdout = capture
    # Reset the stdout to its original value
    sys.stdout = sys.__stdout__
    # Get the captured output as a string
    captured_output = capture.content
    # Get the versions of the libraries
    libraries = [
        #'__future__',
        'lightning',
        'time',
        'torch',
        'torchvision',
        'efficientnet_pytorch',
        'timm',
        'multiprocessing',
        'sklearn',
        'logging',
        'wandb'
    ]
    # Populate with basics 
    basic_log = '%(asctime)s - %(levelname)s - %(message)s\n\n'
    log_format = 'Task name: CBIS-DDSM training for the Reproducibility Experiments\n'
    log_format += f'Run name: {run_name}\n\n'
    log_format += 'Environment:\n'
    log_format += 'Python version: {}\n'.format(platform.python_version())
    log_format += 'PyTorch version: {}\n'.format(torch.__version__)
    for library in libraries:
        try:
            version = pkg_resources.get_distribution(library).version
            log_format += '{} version: {}\n'.format(library, version)
        except pkg_resources.DistributionNotFound:
            log_format += '{} version: Not found\n'.format(library)
    log_format += 'CUDA devices: {}\n\n'.format(torch.cuda.device_count())
    log_format += 'Hardware:\n'
    log_format += 'CPU: {}\n'.format(cpuinfo.get_cpu_info()['brand_raw'])
    # Split the captured output into separate lines
    captured_lines = captured_output.split('\n')
    logging.basicConfig(filename=log_file, level=logging.INFO, format=basic_log)
    logging.info('The task has been started\n')
    log_format += "\n\nSlurm environment variables:\n"
    log_format += f"Config Path: {args.config}\n"
    log_format += f"CUDA Visible Devices: {args.cuda}\n"
    log_format += f"Username: {args.username}\n"
    log_format += f"Job ID: {args.job_id}\n"
    log_format += f"Job Name: {args.job_name}\n"
    log_format += f"Node List: {args.node_list}\n"
    log_format += f"Total Tasks: {args.total_tasks}\n"
    log_format += f"Submit Host: {args.submit_host}\n"
    log_format += f"Current Date: {args.current_date}\n"
    log_format += f"Working Directory: {args.working_directory}\n"
    logging.info(f'Run Infos:\n\n {log_format}')
    device = device
    gpu_log = 'Current GPU information:\n'
    gpu_log+= f'Current GPU device:{args.cuda}\n'
    # Add captured GPU information to the log format line by line
    for i,line in enumerate(captured_lines):
        if i <=7:
            gpu_log += f'{line}\n' 
    logging.info(f'{gpu_log}')
    return device


def get_gpu_info():
    output = subprocess.check_output(['nvidia-smi', '--format=csv,noheader,nounits', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,gpu_bus_id'])
    lines = output.decode().strip().split('\n')

    gpu_info = {}

    process_output = subprocess.check_output(['nvidia-smi', '--format=csv,noheader,nounits', '--query-compute-apps=gpu_bus_id,name'])
    process_lines = process_output.decode().strip().split('\n')

    process_names = {}
    for line in process_lines:
        gpu_bus_id, process_name = line.split(',')
        if gpu_bus_id.strip() in process_names: # means two compute process running on the same gpu
            process_names[gpu_bus_id.strip()] += (' | ' + process_name)
        else:
            process_names[gpu_bus_id.strip()] = process_name


    for line in lines:
        index, name, temperature, utilization, memory_used, memory_total, bus_id = line.split(',')
        index = int(index.strip())
        temperature = int(temperature.strip())
        utilization = int(utilization.strip())
        memory_used = int(memory_used.strip())
        memory_total = int(memory_total.strip())
        bus_id = str(bus_id.strip())

        if str(bus_id) in process_names:
            username = process_names[str(bus_id)]
        else:
            username = 'N/A'

        gpu_info[index] = {'name': name.strip(), 'temperature': temperature, 'utilization': utilization, 'memory_used': memory_used, 'memory_total': memory_total, 'username': username}

    return gpu_info



def log_gpu_info(epoch):
    gpu_info = get_gpu_info()
    table_data = []

    for index, info in gpu_info.items():
        gpu_status = [
            index,
            info['name'],
            f"{info['temperature']}°C",
            f"{info['utilization']}%",
            f"{info['memory_used']} / {info['memory_total']} MB",
            info['username']
        ]
        table_data.append(gpu_status)

    column_names = ['Index', 'Name', 'Temperature', 'Utilization', 'Memory Used/Total', 'Processes']

    log_message = f'\nStart of Epoch: {epoch + 1}\n' 
    log_message += tabulate(table_data, headers=column_names, tablefmt="grid")

    logging.info(log_message)

def log_end_epoch(end_epoch):

    endloss = end_epoch[0]
    accuracy = end_epoch[1]
    f1 = end_epoch[2]
    precision = end_epoch[3]
    recall = end_epoch[4]
    epoch = end_epoch[5]

    # Create the table data
    table_data = [
        ["Metric", "Value"],
        ["Training accuracy", f'{accuracy:.2f}'],
        ["F1 score", f'{f1:.2f}'],
        ["Precision score", f'{precision:.2f}'],
        ["Recall score", f'{recall:.2f}']
    ]
    # Create the log string
    log_string = f'End of epoch {epoch + 1}:\n'
    log_string += f'training loss: {endloss}\n'
    
    # Convert the table to a string
    log_string += tabulate(table_data, tablefmt="grid")
    # Log the table
    logging.info(log_string)

def log_model_info(model):

    # Create the log message
    log_message = "Model Summary:\n"
    for name, param in model.named_parameters():
        if param.requires_grad:
            log_message += f"Layer name: {name}\n"
            log_message += f"Layer shape: {param.shape}\n"
            log_message += f"Layer parameters: {param.numel()}\n"
            log_message += f"Layer trainable: {param.requires_grad}\n"
            log_message += f"Layer dtype: {param.dtype}\n"
            log_message += f"------------------------\n"
            
    log_message += f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n"
    log_message += f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n"
    log_message += f"Total Layers: {len(list(model.parameters()))}\n"
    log_message += f"Total Convolutional Layers: {len([x for x in model.modules() if type(x) == nn.Conv2d])}\n"
    log_message += f"Total Linear Layers: {len([x for x in model.modules() if type(x) == nn.Linear])}\n"


    

    # Log the message
    logging.info(log_message)


def log_results(end_test, elapsed_time):

    acc = end_test[0]
    f1 = end_test[1]
    precision = end_test[2]
    recall = end_test[3]

    # Prepare the results as a table
    headers = ["Metric", "Value"]
    data = [
        ["Accuracy", acc],
        ["F1 Score", f1],
        ["Precision", precision],
        ["Recall", recall]
    ]
    table = '\n'
    table += tabulate(data, headers, tablefmt="grid")

    # Append the total elapsed time
    table += f"\n\nTotal Elapsed Time: {elapsed_time}"

    # Log the results
    logging.info(table)