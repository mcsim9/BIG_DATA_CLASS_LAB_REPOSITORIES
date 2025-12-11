"""
IMDB Data Loader Utility Module
Handles downloading and loading IMDB datasets
"""

import os
import gzip
import requests
import pandas as pd
from pathlib import Path
import time

# IMDB dataset URLs
IMDB_BASE_URL = "https://datasets.imdbws.com/"

DATASETS = {
    'name_basics': 'name.basics.tsv.gz',
    'title_basics': 'title.basics.tsv.gz',
    'title_ratings': 'title.ratings.tsv.gz',
    'title_crew': 'title.crew.tsv.gz',
    'title_akas': 'title.akas.tsv.gz',
    'title_principals': 'title.principals.tsv.gz',
    'title_episode': 'title.episode.tsv.gz'
}

class IMDBDataLoader:
    """Class to handle IMDB data downloading and loading"""
    
    def __init__(self, data_dir='../data'):
        """
        Initialize the data loader
        
        Args:
            data_dir (str): Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_name, force_download=False):
        """
        Download a specific IMDB dataset
        
        Args:
            dataset_name (str): Name of dataset from DATASETS dict
            force_download (bool): If True, download even if file exists
            
        Returns:
            Path: Path to downloaded file
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
        
        filename = DATASETS[dataset_name]
        filepath = self.data_dir / filename
        
        # Check if file already exists
        if filepath.exists() and not force_download:
            print(f"✓ {filename} already exists, skipping download")
            return filepath
        
        url = IMDB_BASE_URL + filename
        print(f"Downloading {filename}...")
        print(f"URL: {url}")
        
        try:
            # Stream download to handle large files
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress indication
            downloaded = 0
            chunk_size = 8192
            start_time = time.time()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Print progress every 10MB
                        if downloaded % (10 * 1024 * 1024) < chunk_size:
                            elapsed = time.time() - start_time
                            speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB) - {speed:.2f} MB/s")
                            else:
                                print(f"  Downloaded: {downloaded / (1024*1024):.1f}MB - {speed:.2f} MB/s")
            
            elapsed = time.time() - start_time
            print(f"✓ Downloaded {filename} ({downloaded / (1024*1024):.1f}MB in {elapsed:.1f}s)")
            return filepath
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error downloading {filename}: {e}")
            print(f"  Please manually download from: {url}")
            print(f"  And place it in: {self.data_dir}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None
    
    def load_dataset(self, dataset_name, nrows=None, usecols=None):
        """
        Load a dataset into a pandas DataFrame
        
        Args:
            dataset_name (str): Name of dataset from DATASETS dict
            nrows (int): Number of rows to read (None for all)
            usecols (list): List of columns to load (None for all)
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        filename = DATASETS[dataset_name]
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            print(f"Attempting to download...")
            filepath = self.download_dataset(dataset_name)
            if filepath is None:
                return None
        
        print(f"Loading {filename}...")
        start_time = time.time()
        
        try:
            # Read compressed TSV file
            # Note: IMDB uses '\N' to represent NULL values
            df = pd.read_csv(
                filepath,
                sep='\t',
                compression='gzip',
                na_values='\\N',
                low_memory=False,
                nrows=nrows,
                usecols=usecols
            )
            
            elapsed = time.time() - start_time
            print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns in {elapsed:.1f}s")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.1f}MB")
            
            return df
            
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
            return None
    
    def load_in_chunks(self, dataset_name, chunksize=100000, usecols=None):
        """
        Load dataset in chunks (generator function)
        Useful for very large datasets that don't fit in memory
        
        Args:
            dataset_name (str): Name of dataset from DATASETS dict
            chunksize (int): Number of rows per chunk
            usecols (list): List of columns to load
            
        Yields:
            pd.DataFrame: Chunk of the dataset
        """
        filename = DATASETS[dataset_name]
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            print(f"Attempting to download...")
            filepath = self.download_dataset(dataset_name)
            if filepath is None:
                return
        
        print(f"Loading {filename} in chunks of {chunksize:,} rows...")
        
        try:
            chunks = pd.read_csv(
                filepath,
                sep='\t',
                compression='gzip',
                na_values='\\N',
                low_memory=False,
                chunksize=chunksize,
                usecols=usecols
            )
            
            chunk_num = 0
            for chunk in chunks:
                chunk_num += 1
                yield chunk
                
        except Exception as e:
            print(f"✗ Error loading chunks from {filename}: {e}")
            return
    
    def download_all(self, exclude=None):
        """
        Download all datasets
        
        Args:
            exclude (list): List of dataset names to exclude
        """
        exclude = exclude or []
        
        print("=" * 60)
        print("Downloading all IMDB datasets")
        print("=" * 60)
        
        for dataset_name in DATASETS.keys():
            if dataset_name not in exclude:
                print(f"\n[{dataset_name}]")
                self.download_dataset(dataset_name)
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
    
    def get_dataset_info(self):
        """
        Get information about available datasets
        
        Returns:
            dict: Information about each dataset
        """
        info = {}
        
        for dataset_name, filename in DATASETS.items():
            filepath = self.data_dir / filename
            info[dataset_name] = {
                'filename': filename,
                'exists': filepath.exists(),
                'size_mb': filepath.stat().st_size / (1024*1024) if filepath.exists() else 0
            }
        
        return info
    
    def print_dataset_info(self):
        """Print formatted information about datasets"""
        info = self.get_dataset_info()
        
        print("\n" + "=" * 70)
        print("IMDB Dataset Information")
        print("=" * 70)
        print(f"{'Dataset':<20} {'Filename':<25} {'Status':<10} {'Size (MB)'}")
        print("-" * 70)
        
        for dataset_name, details in info.items():
            status = "✓ Found" if details['exists'] else "✗ Missing"
            size = f"{details['size_mb']:.1f}" if details['exists'] else "-"
            print(f"{dataset_name:<20} {details['filename']:<25} {status:<10} {size}")
        
        print("=" * 70 + "\n")


def quick_load_sample(dataset_name, nrows=1000, data_dir='../data'):
    """
    Quick function to load a small sample of a dataset
    
    Args:
        dataset_name (str): Name of dataset
        nrows (int): Number of rows to load
        data_dir (str): Data directory
        
    Returns:
        pd.DataFrame: Sample of the dataset
    """
    loader = IMDBDataLoader(data_dir)
    return loader.load_dataset(dataset_name, nrows=nrows)


if __name__ == "__main__":
    # Test the loader
    loader = IMDBDataLoader()
    loader.print_dataset_info()
