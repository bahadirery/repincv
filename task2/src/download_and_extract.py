import os
import requests
import zipfile

from tqdm import tqdm

def download_data(url, file_save_name):
    """
    Download data for a given URL.
    :param url: URL of file to download.
    :param file_save_name: String name to save the file on to disk.
    """

    data_dir = os.path.join('task2', 'inputs')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download the file if not present.
    if not os.path.exists(os.path.join(data_dir, file_save_name)):
        print(f"Downloading {file_save_name}")
        file = requests.get(url, stream=True)
        total_size = int(file.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True
        )
        with open(os.path.join(data_dir, file_save_name), 'wb') as f:
            for data in file.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
    else:
        print('File already present')

def extract_data(file_path):
    # We need to extract the data twice.
    print('Extracting file...')
    
    with zipfile.ZipFile(file_path) as z:
        # Wrap the infolist() with tqdm for progress bar
        for member in tqdm(z.infolist(), desc="Extracting", unit="file"):
            z.extract(member, os.path.join('task2', 'inputs'))
    print("Extracted all")

    # Extract the zip file that has been obtained again.
    with zipfile.ZipFile(os.path.join(
            'task2', 
            'inputs', 
            'DATA_Maguire_20180517_ALL', 
            'SDNET2018.zip'
        )) as z:
        # Wrap the infolist() with tqdm for progress bar
        for member in tqdm(z.infolist(), desc="Extracting", unit="file"):
            z.extract(member, os.path.join('task2', 'inputs', 'SDNET2018'))
    print("Extracted all")



if __name__ == '__main__':
    download_data('https://digitalcommons.usu.edu/context/all_datasets/article/1047/type/native/viewcontent', 'data.zip')
    extract_data(os.path.join('task2', 'inputs', 'data.zip'))