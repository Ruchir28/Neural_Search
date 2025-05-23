import cupy as cp
from typing import Dict, Tuple
import psutil
import os

def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage information.
    
    Returns:
        Dict containing used and total memory in GB for both GPU and pinned memory
    """
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    pinned_total_gb = 0.0
    try:
        pinned_total_gb = pinned_mempool.get_limit() / 1024**3
    except AttributeError:
        # Fallback or warning if get_limit() is also not available
        print("Warning: pinned_mempool.get_limit() not available.")

    return {
        'gpu_used_gb': mempool.used_bytes() / 1024**3,
        'gpu_total_gb': mempool.total_bytes() / 1024**3,
        'pinned_used_gb': 0.0,  # PinnedMemoryPool doesn't have a direct used_bytes() like MemoryPool
        'pinned_total_gb': pinned_total_gb
    }

def get_system_memory_info() -> Dict[str, float]:
    """
    Get current system memory usage information.
    
    Returns:
        Dict containing used and total system memory in GB
    """
    mem = psutil.virtual_memory()
    return {
        'system_used_gb': (mem.total - mem.available) / 1024**3,
        'system_total_gb': mem.total / 1024**3
    }

def print_memory_stats():
    """
    Print formatted memory statistics for both GPU and system memory.
    """
    gpu_info = get_gpu_memory_info()
    sys_info = get_system_memory_info()
    
    print("\n=== Memory Usage Statistics ===")
    
    gpu_percentage = 0.0
    if gpu_info['gpu_total_gb'] > 0:
        gpu_percentage = (gpu_info['gpu_used_gb'] / gpu_info['gpu_total_gb'] * 100)
    print(f"GPU Memory: {gpu_info['gpu_used_gb']:.2f}GB / {gpu_info['gpu_total_gb']:.2f}GB "
          f"({gpu_percentage:.1f}%)")

    # For pinned memory, we might not have a meaningful percentage if used is always 0
    # and total is a limit. So, we can just print used/total without percentage for now.
    # If total is 0 from get_limit(), it will show 0.00GB.
    print(f"Pinned Memory: {gpu_info['pinned_used_gb']:.2f}GB / {gpu_info['pinned_total_gb']:.2f}GB")

    sys_percentage = 0.0
    if sys_info['system_total_gb'] > 0:
        sys_percentage = (sys_info['system_used_gb'] / sys_info['system_total_gb'] * 100)
    print(f"System Memory: {sys_info['system_used_gb']:.2f}GB / {sys_info['system_total_gb']:.2f}GB "
          f"({sys_percentage:.1f}%)")
    print("=============================\n") 

if __name__ == "__main__":
    print_memory_stats() 