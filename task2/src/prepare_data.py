import pandas as pd
import os
import glob as glob

from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

ROOT_DIR = os.path.join('task2', 'inputs', 'SDNET2018')

# Get all the image folder paths. Structure as such so that the files inside
# the folders in `all_paths` should contain the images.
all_paths = glob.glob(os.path.join(ROOT_DIR, '*', '*'), recursive=True)
folder_paths = [x for x in all_paths if os.path.isdir(x)]
print(f"Folder paths: {folder_paths}")
print(f"Number of folders: {len(folder_paths)}")

# The class names. We will consider 0 as 'Crack Detected' 
# and 1 as 'Crack Undetected'. 0 means image folder starting with 'C', 
# as in 'CW' and 1 means image folder starting with 'U' as in 'UW'.

# Create a DataFrame
data = pd.DataFrame()

# Image formats to consider.
image_formats = ['jpg', 'JPG', 'PNG', 'png']
labels = []
counter = 0
for i, folder_path in tqdm(enumerate(folder_paths), total=len(folder_paths)):
    image_paths = os.listdir(folder_path)
    folder_name = folder_path.split(os.path.sep)[-1]
    if folder_name.startswith('C'):
        label = 0
    if folder_name.startswith('U'):
        label = 1
    # Save image paths in the DataFrame.
    for image_path in image_paths:
        if image_path.split('.')[-1] in image_formats:
            path_to_save = os.path.join(folder_path, image_path)
            data.loc[counter, 'image_path'] = path_to_save
            data.loc[counter, 'target'] = int(label)
            labels.append(label)
            counter += 1

# Shuffle the dataset.
data = data.sample(frac=1).reset_index(drop=True)

# Data to be used for training and validation.
trainval_split = 0.9

total_instances = len(data)
trainval_instances = int(total_instances*trainval_split)
test_instances = total_instances - trainval_instances 

print(f"Training and validation instances: {trainval_instances}")
print(f"Test instances: {test_instances}")
# Save as CSV file
data.iloc[:trainval_instances].to_csv(os.path.join('task2', 'inputs', 'trainval.csv'), index=False)
data.iloc[trainval_instances:].to_csv(os.path.join('task2', 'inputs', 'test.csv'), index=False)