"""
Data download script for Visual Question Answering project.
Downloads a subset of the CLEVR dataset for initial experimentation.
"""

import os
import requests
import tarfile
from tqdm import tqdm

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# CLEVR dataset URLs (using a small subset for initial testing)
CLEVR_URLS = {
    'images': 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip',
    'questions': 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip'
}

def download_file(url, destination):
    """Download a file with progress bar."""
    print(f"Downloading {url} to {destination}")
    
    # Stream the download to handle large files
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_tar(file_path, extract_to):
    """Extract a tar file."""
    print(f"Extracting {file_path} to {extract_to}")
    with tarfile.open(file_path) as tar:
        tar.extractall(path=extract_to)

if __name__ == "__main__":
    print("Starting data download...")
    
    # Download the dataset
    zip_path = os.path.join('data', 'CLEVR_v1.0.zip')
    download_file(CLEVR_URLS['images'], zip_path)
    
    # Extract the dataset
    extract_tar(zip_path, 'data')
    
    print("Download complete! Data available in data/CLEVR_v1.0/")
    print("\nNext steps:")
    print("1. Explore the dataset structure")
    print("2. Begin data exploration in notebooks/01_data_exploration.ipynb")