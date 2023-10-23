import subprocess
import time
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Run tasks with configurations.")
parser.add_argument('--tasks', nargs='+', choices=['task1', 'task2', 'task3'], default=['task1', 'task2', 'task3'],
                    help="Specify which tasks to run. Options: 'task1', 'task2', 'task3'. Default is all.")
args = parser.parse_args()

### TASK 1 ###

if 'task1' in args.tasks:
    # Base command
    base_cmd = ["python3", "task1/src/main.py", "--config"]

    # Loop through config files
    for i in range(1, 111):  # Adjusted range to go up to 110
        # Adjusted format string to handle up to three digits
        config_file = f"task1/configs/task1/cfg_{i:03}.yaml"
        cmd = base_cmd + [config_file]
        
        # Print the current config file and timestamp
        start_time = time.time()
        print(f"Executing with config file: {config_file}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        # Run the command
        subprocess.run(cmd)
        
        # Calculate and print the duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Execution time for {config_file}: {duration:.2f} seconds")
        print("-" * 50)  # Separator for clarity

### TASK 2 ###

if 'task2' in args.tasks:
    # Run the initial setup scripts once
    setup_scripts = [
        "task2/src/download_and_extract.py",
        "task2/src/prepare_data.py"
    ]

    for script in setup_scripts:
        print(f"Executing setup script: {script}")
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        subprocess.run(["python3", script])
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Execution time for {script}: {duration:.2f} seconds")
        print("-" * 50)  # Separator for clarity

    # Base command
    base_cmd = ["python3", "task2/src/train.py", "--config"]

    # Loop through config files
    for i in range(1, 111):
        config_file = f"task2/configs/task2/cfg_{i:03}.yaml"
        cmd = base_cmd + [config_file]
        
        # Print the current config file and timestamp
        start_time = time.time()
        print(f"Executing with config file: {config_file}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        # Run the command
        subprocess.run(cmd)
        
        # Calculate and print the duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Execution time for {config_file}: {duration:.2f} seconds")
        print("-" * 50)  # Separator for clarity


### TASK 3 ###

if 'task3' in args.tasks:
    # Base command
    base_cmd = ["python3", "task3/src/train_eval/train.py"]

    # Loop through config files
    for i in range(1, 61):
        config_file_path = f"task3/experiment_setups/setup_{i:02}/config_8_run_4.ini"
        config = f"task3/experiment_setups/setup_{i:02}/cfg_{i:02}.yaml"
        
        cmd = base_cmd + ["--config_file_path", config_file_path, "--config", config]
        
        # Print the current config file and timestamp
        start_time = time.time()
        print(f"Executing with config: {config}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        # Run the command
        subprocess.run(cmd)
        
        # Calculate and print the duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Execution time for config: {config}: {duration:.2f} seconds")
        print("-" * 50)  # Separator for clarity
