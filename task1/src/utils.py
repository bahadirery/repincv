'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
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
import gpustat
import subprocess
import os
import torch
import random
import numpy as np
from time import time, strftime, localtime
from tabulate import tabulate
import torch.nn as nn
from torchsummary import summary
import contextlib
import torch_random


try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except:
    pass

TOTAL_BAR_LENGTH = 65.
last_time = time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
# show me an example class syntax code in python
class ExampleClass: 
    def __init__(self, name, age): 
        self.name = name 
        self.age = age 
    def show(self): 
        print(self.name) 
        print(self.age)

             
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
        torch_random.manual_seed(self.seed)  # It calls cuda seed internally if cuda is available. But I don't use cuda here.

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
        np.random.seed(self.seed)
        random.seed(self.seed)
        print(f"Worker {worker_id} seeded with {self.seed}")
    



def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def create_wandb_plot(x_data=None, y_data=None, x_name=None, y_name=None, title=None, log=None, plot="line"):
    '''Create and save lineplot/barplot in W&B Environment.
    x_data & y_data: Pandas Series containing x & y data
    x_name & y_name: strings containing axis names
    title: title of the graph
    log: string containing name of log'''
    
    data = [[label, val] for (label, val) in zip(x_data, y_data)]
    table = wandb.Table(data=data, columns = [x_name, y_name])
    
    if plot == "line":
        wandb.log({log : wandb.plot.line(table, x_name, y_name, title=title)})
    elif plot == "bar":
        wandb.log({log : wandb.plot.bar(table, x_name, y_name, title=title)})
    elif plot == "scatter":
        wandb.log({log : wandb.plot.scatter(table, x_name, y_name, title=title)})

def create_wandb_hist(x_data=None, x_name=None, title=None, log=None):
    '''Create and save histogram in W&B Environment.
    x_data: Pandas Series containing x values
    x_name: strings containing axis name
    title: title of the graph
    log: string containing name of log'''
    
    data = [[x] for x in x_data]
    table = wandb.Table(data=data, columns=[x_name])
    wandb.log({log : wandb.plot.histogram(table, x_name, title=title)})


def log_basics(log_filename,output_base,run_name,device,args):


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
    log_format = 'Task name: CIFAR-10 training for the Reproducibility Experiments\n'
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
    # Capture the output
    log_stream = io.StringIO()
    with contextlib.redirect_stdout(log_stream):
        summary(model, input_size=(3, 32, 32), device="cpu")

    # Store the captured output as a string
    log_message += log_stream.getvalue()

    input_size = (3, 32, 32)
    input_tensor = torch.randn(1, *input_size)
    with torch.no_grad():
        output = model(input_tensor)
    output_size = tuple(output.size()[1:])
    log_message += f"Input Size: {input_size}\n"
    log_message += f"Output Size: {output_size}\n"

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