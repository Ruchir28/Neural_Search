import numpy as np
from datasets import load_dataset
import os

DATASET = "Cohere/wikipedia-22-12-en-embeddings"
CACHE_DIR = "dataset_cache"
# ROW2LIST_PATH = "lists_info/ivf_pq_layout.npz" # No longer needed
ID_TO_SORTED_ROW_PATH = "lists_info/vec_id_to_sorted_row.bin.npy" # Source of truth for ordering
OUTPUT_PATH = "lists_info/docs_sorted.parquet"
COLUMNS_TO_KEEP = ["title", "text", "url"]
SORT_KEY_COLUMN = "_final_sort_order_key"
OUT_DIR = "lists_info/meta_sorted"

def main():
    # Load the mapping from original document ID (index) to its final sorted row index
    print(f"Loading id_to_sorted_row map from {ID_TO_SORTED_ROW_PATH}...")
    id_map_orig_to_sorted = np.load(ID_TO_SORTED_ROW_PATH, mmap_mode="r")
    print(f"Loaded id_map_orig_to_sorted of length {len(id_map_orig_to_sorted)}")

    original_dataset = load_dataset(DATASET, split="train", cache_dir=CACHE_DIR)
    print(f"Loaded original dataset with {len(original_dataset)} rows")

    if len(original_dataset) != len(id_map_orig_to_sorted):
        raise ValueError(
            f"Length mismatch: original dataset has {len(original_dataset)} rows, "
            f"but id_map_orig_to_sorted has {len(id_map_orig_to_sorted)} entries. "
            f"These must match to correctly order the metadata."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Selecting columns from original dataset...")
    dataset_with_cols = original_dataset.select_columns(COLUMNS_TO_KEEP)
    
    print(f"Adding sort key column '{SORT_KEY_COLUMN}'...")
    # The id_map_orig_to_sorted array provides the target sorted row index for each original document index.
    # This array itself becomes the sort key.
    dataset_with_key = dataset_with_cols.add_column(name=SORT_KEY_COLUMN, column=id_map_orig_to_sorted)
    
    print("Sort key column added.")

    print(f"Sorting dataset by '{SORT_KEY_COLUMN}'...")
    sorted_dataset = dataset_with_key.sort(SORT_KEY_COLUMN)
    print("Dataset sorted.")

    print("Removing temporary sort key column...")
    sorted_dataset = sorted_dataset.remove_columns([SORT_KEY_COLUMN])
    print("Sort key column removed.")
    
    print("Flattening indices of the sorted dataset...")
    sorted_dataset = sorted_dataset.flatten_indices(
        num_proc=16,
    )
    print("Indices flattened.")
    
    sorted_dataset.save_to_disk(OUT_DIR)

if __name__ == "__main__":
    main() 
