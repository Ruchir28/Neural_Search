import cupy as cp
import numpy as np
import time
from pathlib import Path
from datasets import load_dataset
from cuvs.neighbors import ivf_pq
from tqdm import tqdm
import os
from utils.gpu_memory_monitor import print_memory_stats

# Configuration
DATASET = "Cohere/wikipedia-22-12-en-embeddings"
CACHE_DIR = "dataset_cache"

# Training Parameters
N_TRAIN = 2_000_000  # Number of vectors for training
BATCH_SIZE = 250_000  # Increased batch size for better performance
VECTOR_DIM = 768     # Dimension of embeddings

# Index Parameters
N_LISTS = 32768  # Number of IVF clusters
M = 96  # Number of PQ sub-quantizers (768/96 = 8 dims per subquantizer)
BITS_PER_CODE = 8  # Bits per PQ code (256 centroids per subquantizer)


def prepare_training_data(n_train: int = N_TRAIN, batch_size: int = BATCH_SIZE):
    """
    Load and prepare training data efficiently using direct RAM allocation
    """
    print("\nInitializing dataset...")
    dataset = load_dataset(DATASET, split=f"train[:{n_train}]", cache_dir=CACHE_DIR)

    def extract_emb(batch):
        return { "emb": batch["emb"] }

    dataset = dataset.map(
        extract_emb,
        batched=True,
        batch_size=10000,  # Batch size for the .map() operation
        num_proc=os.cpu_count(),
        remove_columns=[col for col in dataset.column_names if col != "emb"]
    )

    # Now, convert the processed dataset (which only has 'emb') to NumPy batch by batch
    print(f"\nPre-allocating NumPy array for {n_train} vectors of dim {VECTOR_DIM} with dtype float16...")
    train_vecs = np.zeros((n_train, VECTOR_DIM), dtype=np.float16)
    
    print("\nFilling NumPy array in batches...")
    loaded_count = 0
    # Define a batch size for this iteration step.
    iteration_batch_size = 100_000 
    
    with tqdm(total=n_train, desc="Converting to NumPy") as pbar:
        # Iterate over the dataset (which now only contains the 'emb' column)
        for batch in dataset.iter(batch_size=iteration_batch_size):
            # batch is a dictionary like {'emb': [embedding1, embedding2, ...]}
            embeddings_in_batch = np.array(batch["emb"], dtype=np.float16)
            
            current_batch_actual_size = embeddings_in_batch.shape[0]
            
            # Ensure we don't write past the allocated space if dataset.iter yields more than n_train
            # This should ideally not be an issue if dataset was sliced correctly with train[:n_train]
            write_size = min(current_batch_actual_size, n_train - loaded_count)
            if write_size <= 0: # No more space or already filled n_train
                break

            train_vecs[loaded_count : loaded_count + write_size] = embeddings_in_batch[:write_size]
            loaded_count += write_size
            pbar.update(write_size)

            if loaded_count >= n_train:
                break
            
    if loaded_count < n_train:
        print(f"\nWarning: Loaded {loaded_count} vectors, but expected {n_train}. The array might be partially empty or smaller than expected.")
        # Optionally, truncate if fewer than n_train vectors were actually loaded and are valid
        # train_vecs = train_vecs[:loaded_count]
        # Or, if it's critical to have exactly n_train and it's an error, raise an exception.

    print(f"\nSuccessfully converted {loaded_count} embeddings to NumPy array.")
  
    return train_vecs

def train_index(train_vecs: np.ndarray):
    """
    Train the IVF-PQ index using cuVS
    """
    print("\nTransferring training vectors to GPU...")
    train_vecs_gpu = cp.asarray(train_vecs)
    print_memory_stats()
    
    # Define index parameters
    print("\nConfiguring index parameters...")
    index_params = ivf_pq.IndexParams(
        n_lists=N_LISTS,
        metric="inner_product",  # Using inner product for normalized vectors
        pq_dim=M,  # M is your number of PQ sub-quantizers (segments)
        pq_bits=BITS_PER_CODE,
        kmeans_trainset_fraction=1.0  # Use all training vectors for k-means
    )
    
    # Train the index with proper stream handling
    print("\nTraining index...")
    trained_index = ivf_pq.build(
        index_params,
        train_vecs_gpu,
    )
    
    print_memory_stats()
    return trained_index

def save_trained_index(trained_index: ivf_pq.Index, output_path: str):
    """
    Save the trained (empty) index for later population
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the index
    ivf_pq.save(str(output_path), trained_index)

def main():
    # Setup output directory
    output_dir = Path("trained_indices")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Define the standard output path for the trained index
    output_path = output_dir / "trained_ivfpq_index.bin"

    # 1. Prepare training data
    print("Loading training data for index construction...")
    data_prep_start_time = time.time()
    train_vecs = prepare_training_data()
    data_prep_time = time.time() - data_prep_start_time
    print(f"\nData preparation completed in {data_prep_time:.2f} seconds")
    
    # 2. Train index
    print("\nTraining IVF-PQ index...")
    print_memory_stats()
    train_start_time = time.time()
    trained_index = train_index(train_vecs)
    train_time = time.time() - train_start_time
    print(f"\nIndex training completed in {train_time:.2f} seconds")
    print_memory_stats()
    
    # 3. Save trained index
    print(f"\nSaving trained index to {output_path}...")
    save_trained_index(trained_index, str(output_path))
    
    print(f"\nTraining pipeline completed. Index saved to {output_path}")
    total_time = data_prep_time + train_time
    print(f"Total training time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 