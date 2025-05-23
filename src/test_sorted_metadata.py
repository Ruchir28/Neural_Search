import numpy as np
from datasets import load_dataset, load_from_disk
import logging

# --- Configuration ---
METADATA_PARQUET_PATH = "lists_info/docs_sorted.parquet"
ID_TO_SORTED_ROW_PATH = "lists_info/vec_id_to_sorted_row.bin.npy"
# Ensure these match the parameters used in create_metadata_parquet.py
ORIGINAL_DATASET_NAME = "Cohere/wikipedia-22-12-en-embeddings"
CACHE_DIR = "dataset_cache"
COLUMNS_TO_COMPARE = ["title", "text", "url"] # Columns created by create_metadata_parquet.py
NUM_SAMPLES_TO_TEST = 10 # Number of random samples to check
SORTED_METADATA_PATH = "lists_info/meta_sorted"
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("--- Starting Metadata Parquet Test ---")
    logging.info("Loading data...")

    try:
        # Load Parquet metadata
        # parquet_data = load_dataset("parquet", data_files=METADATA_PARQUET_PATH, split="train", cache_dir=CACHE_DIR)
        parquet_data = load_from_disk(SORTED_METADATA_PATH)
        logging.info(f"Successfully loaded Parquet metadata from '{SORTED_METADATA_PATH}' with {len(parquet_data)} rows.")

        # Load the original_id to sorted_row mapping
        id_to_sorted_row = np.load(ID_TO_SORTED_ROW_PATH, mmap_mode="r")
        logging.info(f"Successfully loaded id_to_sorted_row map from '{ID_TO_SORTED_ROW_PATH}' with {len(id_to_sorted_row)} entries.")

        # Load the original dataset for comparison
        original_dataset = load_dataset(ORIGINAL_DATASET_NAME, split="train", cache_dir=CACHE_DIR)
        logging.info(f"Successfully loaded original dataset '{ORIGINAL_DATASET_NAME}' with {len(original_dataset)} rows.")

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # --- Sanity Checks ---
    if len(id_to_sorted_row) != len(parquet_data):
        logging.warning(f"Length mismatch! id_to_sorted_row has {len(id_to_sorted_row)} elements, "
                        f"but docs_sorted.parquet has {len(parquet_data)} rows. "
                        "This indicates a potential inconsistency in the data generation pipeline. "
                        "Test results might be compromised.")
    
    if len(original_dataset) != len(id_to_sorted_row):
         logging.info(f"Length of original dataset ({len(original_dataset)}) does not match "
                      f"length of id_to_sorted_row map ({len(id_to_sorted_row)}). "
                      "This is acceptable if the indexing process intentionally selected a subset of the original dataset.")

    logging.info(f"--- Test Setup: Comparing {NUM_SAMPLES_TO_TEST} random samples ---")
    
    # Determine the valid range for original IDs based on the smallest relevant dataset/map
    max_id_for_sampling = min(len(original_dataset), len(id_to_sorted_row)) - 1

    if max_id_for_sampling < 0:
        logging.error("Cannot select sample IDs: original dataset or id_map is effectively empty or too small.")
        return
    
    if NUM_SAMPLES_TO_TEST > (max_id_for_sampling + 1) :
        actual_samples_to_test = max_id_for_sampling + 1
        logging.warning(f"Requested NUM_SAMPLES_TO_TEST ({NUM_SAMPLES_TO_TEST}) is greater than available unique IDs ({actual_samples_to_test}). "
                        f"Testing all {actual_samples_to_test} available IDs instead.")
    else:
        actual_samples_to_test = NUM_SAMPLES_TO_TEST

    if actual_samples_to_test == 0:
        logging.info("No samples to test based on available data.")
        return

    sample_original_ids = np.random.choice(np.arange(max_id_for_sampling + 1), 
                                           size=actual_samples_to_test, 
                                           replace=False)

    all_samples_match = True
    mismatched_samples = 0

    for i, original_id_val in enumerate(sample_original_ids):
        original_id = int(original_id_val) # Ensure it's a Python int
        logging.info(f"--- Test Case {i+1}/{actual_samples_to_test} ---")
        logging.info(f"Original Document ID (from original_dataset index): {original_id}")

        current_sample_match = True
        try:
            # 1. Get the corresponding row index in the sorted data structures
            if original_id >= len(id_to_sorted_row):
                logging.error(f"  Original ID {original_id} is out of bounds for id_to_sorted_row (len: {len(id_to_sorted_row)}).")
                all_samples_match = False
                mismatched_samples +=1
                continue
            sorted_row_index = int(id_to_sorted_row[original_id])
            logging.info(f"  Mapped Sorted Row Index (for Parquet & sorted embeddings): {sorted_row_index}")

            # 2. Fetch metadata from the Parquet file using the sorted_row_index
            if sorted_row_index >= len(parquet_data):
                logging.error(f"  Sorted Row Index {sorted_row_index} is out of bounds for Parquet data (len: {len(parquet_data)}).")
                all_samples_match = False
                mismatched_samples +=1
                continue
            metadata_from_parquet = parquet_data[sorted_row_index]
            
            # 3. Fetch metadata from the original dataset using the original_id
            # original_id is already checked against len(original_dataset) by max_id_for_sampling logic
            metadata_from_original = original_dataset[original_id]

            logging.info("  Comparing metadata fields...")
            for col in COLUMNS_TO_COMPARE:
                val_parquet = metadata_from_parquet.get(col)
                val_original = metadata_from_original.get(col)
                
                # Simple string comparison. For more complex data, enhance this.
                if str(val_parquet) != str(val_original):
                    logging.warning(f"    MISMATCH in column '{col}':")
                    logging.warning(f"      Parquet  (type {type(val_parquet)}): '{str(val_parquet)[:100]}{'...' if len(str(val_parquet)) > 100 else ''}'")
                    logging.warning(f"      Original (type {type(val_original)}): '{str(val_original)[:100]}{'...' if len(str(val_original)) > 100 else ''}'")
                    current_sample_match = False
                # else:
                #     logging.info(f"    MATCH in column '{col}'") # Can be verbose
            
            if current_sample_match:
                logging.info(f"  SUCCESS: Metadata for original_id {original_id} (sorted_row {sorted_row_index}) matches.")
            else:
                logging.error(f"  FAILURE: Metadata for original_id {original_id} (sorted_row {sorted_row_index}) does NOT match.")
                all_samples_match = False
                mismatched_samples += 1

        except Exception as e:
            logging.error(f"  EXCEPTION during test for original_id {original_id}: {e}", exc_info=True)
            all_samples_match = False
            mismatched_samples += 1
            continue

    logging.info("--- Test Summary ---")
    if all_samples_match:
        logging.info(f"SUCCESS: All {actual_samples_to_test} tested samples showed matching metadata.")
    else:
        logging.error(f"FAILURE: {mismatched_samples}/{actual_samples_to_test} samples showed metadata mismatches or errors. Please review logs.")

if __name__ == "__main__":
    main() 