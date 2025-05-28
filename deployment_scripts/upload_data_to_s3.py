#!/usr/bin/env python3
"""
Semantic Search Data Upload Script
This script uploads data files directly to S3.
"""

import os
import sys
import time
from datetime import datetime
import threading

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
except ImportError:
    print("ERROR: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)

# Configuration
BUCKET_NAME = f"semantic-search-data-{datetime.now().strftime('%Y%m%d%H%M%S')}"
DEPLOYMENT_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
S3_PREFIX = f"deployments/v{DEPLOYMENT_VERSION}"

# Required data files
REQUIRED_DATA_FILES = [
    "trained_indices/trained_ivfpq_index_full.bin",
    "lists_info/embeddings_listwise.memmap",
    "lists_info/layout_sorted.npz",
    "lists_info/vec_id_to_sorted_row.bin.npy",
    "lists_info/metadata_25gb.lmdb"
]

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    NC = '\033[0m'

def print_status(message: str):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {message}")

def print_error(message: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

class ProgressPercentage:
    """Progress callback for S3 uploads"""
    
    def __init__(self, filename: str, file_size: int):
        self._filename = filename
        self._size = file_size
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._start_time = time.time()
        
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            elapsed = time.time() - self._start_time
            
            if elapsed > 0:
                speed_mbps = (self._seen_so_far / (1024 * 1024)) / elapsed
                eta_seconds = (self._size - self._seen_so_far) / (self._seen_so_far / elapsed) if self._seen_so_far > 0 else 0
                eta_minutes = eta_seconds / 60
                
                sys.stdout.write(
                    f"\r{Colors.PURPLE}  {self._filename}: "
                    f"{percentage:.1f}% "
                    f"({self._seen_so_far / (1024*1024):.1f}MB / {self._size / (1024*1024):.1f}MB) "
                    f"Speed: {speed_mbps:.1f} MB/s "
                    f"ETA: {eta_minutes:.1f}m{Colors.NC}"
                )
                sys.stdout.flush()

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            print_error("AWS credentials not found. Please configure with 'aws configure'")
            return False
        return True
    except Exception as e:
        print_error(f"Error checking AWS credentials: {e}")
        return False

def check_required_files():
    """Check if all required data files exist"""
    missing_files = []
    for file_path in REQUIRED_DATA_FILES:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    return missing_files

def upload_file_to_s3(s3_client, file_path: str, bucket: str, key: str):
    """Upload a single file to S3 with progress tracking"""
    file_size = os.path.getsize(file_path)
    filename = os.path.basename(file_path)
    
    if file_size > 100 * 1024 * 1024:  # Show progress for files > 100MB
        progress = ProgressPercentage(filename, file_size)
        s3_client.upload_file(file_path, bucket, key, Callback=progress)
        print()  # New line after progress
    else:
        s3_client.upload_file(file_path, bucket, key)
        print_status(f"Uploaded: {filename}")

def upload_directory_to_s3(s3_client, local_dir: str, bucket: str, s3_prefix: str):
    """Upload entire directory to S3"""
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")
            
            upload_file_to_s3(s3_client, local_path, bucket, s3_key)

def create_s3_client():
    """Create S3 client with optimized configuration"""
    config = Config(
        region_name='us-east-1',
        retries={'max_attempts': 3, 'mode': 'adaptive'},
        max_pool_connections=50,
        s3={
            'multipart_threshold': 64 * 1024 * 1024,  # 64MB
            'multipart_chunksize': 16 * 1024 * 1024,  # 16MB
            'max_concurrency': 10,
        }
    )
    return boto3.client('s3', config=config)

def main():
    """Main execution function"""
    print(f"{Colors.BLUE}=== Semantic Search Data Upload ==={Colors.NC}")
    print(f"{Colors.YELLOW}Version: {DEPLOYMENT_VERSION}{Colors.NC}")
    print(f"{Colors.YELLOW}S3 Bucket: {BUCKET_NAME}{Colors.NC}")
    print(f"{Colors.YELLOW}S3 Prefix: {S3_PREFIX}{Colors.NC}")
    print()
    
    # Check AWS credentials
    if not check_aws_credentials():
        sys.exit(1)
    
    # Check required files
    missing_files = check_required_files()
    if missing_files:
        print_error("Missing required data files:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)
    
    print_status("All required data files found!")
    
    # Create S3 client
    s3_client = create_s3_client()
    
    # Create bucket if it doesn't exist
    print_status("Checking/creating S3 bucket...")
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print_status(f"S3 bucket exists: {BUCKET_NAME}")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print_status(f"Creating S3 bucket: {BUCKET_NAME}")
            # For regions other than us-east-1, need to specify LocationConstraint
            if 'us-west-1' != 'us-east-1':
                s3_client.create_bucket(
                    Bucket=BUCKET_NAME,
                    CreateBucketConfiguration={'LocationConstraint': 'us-west-1'}
                )
            else:
                s3_client.create_bucket(Bucket=BUCKET_NAME)
        else:
            print_error(f"Error checking bucket: {e}")
            sys.exit(1)
    
    # Upload data files directly to S3
    print_status("Starting direct upload to S3...")
    print(f"{Colors.BLUE}This may take 15-30 minutes for large files...{Colors.NC}")
    
    start_time = time.time()
    
    try:
        # Upload directories directly
        print_status("Uploading trained_indices/...")
        upload_directory_to_s3(s3_client, "trained_indices", BUCKET_NAME, f"{S3_PREFIX}/trained_indices")
        
        print_status("Uploading lists_info/...")
        upload_directory_to_s3(s3_client, "lists_info", BUCKET_NAME, f"{S3_PREFIX}/lists_info")
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print()
        print(f"{Colors.GREEN}=== Upload Complete! ==={Colors.NC}")
        print(f"{Colors.BLUE}S3 Location:{Colors.NC} s3://{BUCKET_NAME}/{S3_PREFIX}")
        print(f"{Colors.BLUE}Upload Time:{Colors.NC} {elapsed_time/60:.1f} minutes")
        
    except Exception as e:
        print_error(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 