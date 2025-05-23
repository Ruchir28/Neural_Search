import cupy as cp
import numpy as np
from pathlib import Path
from cuvs.neighbors import ivf_pq
from train_index import prepare_training_data, N_TRAIN, VECTOR_DIM, DATASET, CACHE_DIR
from datasets import load_dataset
from tqdm import tqdm

# --- Configurable parameters ---
INDEX_PATH = Path("trained_indices/trained_ivfpq_index.bin")
OUTPUT_INDEX_PATH = Path("trained_indices/trained_ivfpq_index_full.bin")
BATCH_SIZE = 100_000  # Number of embeddings to add per batch (adjust as needed)

# --- Load the trained index ---
print(f"Loading trained index from {INDEX_PATH}...")
index = ivf_pq.load(str(INDEX_PATH))
print("Index loaded.")

# --- Load the dataset (using HuggingFace datasets as in train_index) ---
print(f"Loading dataset metadata: {DATASET} ...")
dataset = load_dataset(DATASET, split="train", cache_dir=CACHE_DIR)
total_embeddings = len(dataset)
print(f"Total embeddings in dataset: {total_embeddings}")

# Use numpy format for efficient batch extraction
# This ensures that slicing returns numpy arrays for columns
print("Setting dataset format to numpy for efficient batch extraction...")
dataset = dataset.with_format("numpy")

# --- Add remaining embeddings in batches ---
start_idx = N_TRAIN
end_idx = total_embeddings
num_to_add = end_idx - start_idx
print(f"Adding embeddings from {start_idx} to {end_idx-1} (total: {num_to_add}) in batches of {BATCH_SIZE}")

for batch_start in tqdm(range(start_idx, end_idx, BATCH_SIZE), desc="Adding batches"):
    batch_end = min(batch_start + BATCH_SIZE, end_idx)
    # Use slicing for efficient batch extraction
    batch_data = dataset[batch_start:batch_end]
    batch_vecs = batch_data["emb"]  # Already a numpy array due to with_format
    # Move to GPU
    batch_vecs_gpu = cp.asarray(batch_vecs)
    # indices for the batch
    ids_gpu = cp.arange(batch_start,
                        batch_start + len(batch_vecs),
                        dtype=cp.int64)
    # Add to index
    ivf_pq.extend(index, batch_vecs_gpu, ids_gpu)
    # Free GPU memory
    del batch_vecs_gpu, ids_gpu
    cp.get_default_memory_pool().free_all_blocks()
    print(f"Added embeddings {batch_start} to {batch_end-1}")

print("All batches added. Saving updated index...")
ivf_pq.save(str(OUTPUT_INDEX_PATH), index)
print(f"Index saved to {OUTPUT_INDEX_PATH}") 