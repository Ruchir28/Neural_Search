import numpy as np
from datasets import load_dataset, load_from_disk
import os
import lmdb
import struct

DATASET = "Cohere/wikipedia-22-12-en-embeddings"  # Keep for reference
CACHE_DIR = "dataset_cache"
SORTED_PARQUET_DIR = "lists_info/meta_sorted"  # Directory containing sorted Parquet files
LMDB_META_OUTPUT_PATH = "lists_info/metadata.lmdb"

# Estimate map size: ~35M records for Cohere/wikipedia-22-12.
# Avg record size (binary format with title, text, url)? 
# Typically smaller than JSON, but conservatively estimate 2-4KB per record for safety.
# 35M * 4KB = ~140GB. Set to 150GB for headroom.
LMDB_MAP_SIZE = 150 * 1024**3  # 150 GiB for 35M records


def write_metadata_to_lmdb_binary_format(
    sorted_dataset,  # Now takes already sorted dataset
    lmdb_path, 
    map_size,
    db_name=b'metadata',
    batch_size=50000  # Batch size can be tuned
):
    """
    Writes dataset metadata to LMDB using:
    - Key: The document's position in the sorted order (0, 1, 2...) as an 8-byte integer
    - Value: Binary blob with [text_len(4)][title_len(4)][url_len(4)][text][title][url]
    
    This approach maintains physical grouping of documents based on their sorted order,
    which typically follows IVF list organization. When documents from the same IVF list
    are accessed together, they benefit from sequential reads and page cache efficiency.
    
    The lookup flow:
    1. Vector search returns original document IDs
    2. Map these to sorted positions using vector_id_to_sorted_row
    3. Use those positions as keys to fetch metadata from LMDB
    """
    print(f"Opening LMDB environment at {lmdb_path}...")
    env = lmdb.open(lmdb_path, map_size=map_size, max_dbs=2, writemap=True, map_async=True)
    
    num_records = len(sorted_dataset)
    print(f"Preparing to write {num_records} records to LMDB using binary format.")

    # Debug: Print information about the dataset structure
    print("Dataset features:", sorted_dataset.features)
    print("First record format:", type(sorted_dataset[0]))
    first_record = sorted_dataset[0]
    print("First record content:", first_record)
    
    try:
        # Open the named database
        with env.begin(write=True) as txn:
            metadata_db = env.open_db(db_name, txn=txn, create=True)
        
        # Process in batches using the dataset's select method
        for batch_start in range(0, num_records, batch_size):
            with env.begin(write=True) as txn_batch:
                batch_end = min(batch_start + batch_size, num_records)
                
                # Get batch using dataset's select method
                batch_indices = range(batch_start, batch_end)
                batch = sorted_dataset.select(batch_indices)
                
                # Process each record in the batch
                for i, record in enumerate(batch):
                    current_idx = batch_start + i
                    
                    # The key is just the position in the sorted dataset
                    key = struct.pack("!Q", current_idx)
                    
                    # Encode metadata value in binary format
                    text_bytes = str(record['text']).encode('utf-8')
                    title_bytes = str(record['title']).encode('utf-8')
                    url_bytes = str(record['url']).encode('utf-8')
                    
                    # Format: [text_len(4)][title_len(4)][url_len(4)][text][title][url]
                    header = struct.pack("!III", len(text_bytes), len(title_bytes), len(url_bytes))
                    value = header + text_bytes + title_bytes + url_bytes
                    
                    # Store in LMDB
                    txn_batch.put(key, value, db=metadata_db)
                
                processed_records = batch_end
                print(f"Committing batch to LMDB: {processed_records}/{num_records} records written.")
            
        print("LMDB write complete.")
    except Exception as e:
        print(f"Error during LMDB write: {e}")
        print(f"Error occurred at record index: {batch_start + i if 'i' in locals() else 'unknown'}")
        if 'record' in locals():
            print(f"Problematic record type: {type(record)}")
            print(f"Problematic record content: {record}")
        raise
    finally:
        env.close()
        print("LMDB environment closed.")


def main():
    # Load the already sorted dataset from Parquet
    print(f"Loading sorted dataset from {SORTED_PARQUET_DIR}...")
    sorted_dataset = load_from_disk(SORTED_PARQUET_DIR)
    print(f"Loaded sorted dataset with {len(sorted_dataset)} rows")
    
    # Debug: Print dataset information
    print("Dataset format:", sorted_dataset.format)
    print("Dataset features:", sorted_dataset.features)
    print("Dataset columns:", sorted_dataset.column_names)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(LMDB_META_OUTPUT_PATH), exist_ok=True)
    
    # Write to LMDB with binary format and sequential keys
    print(f"Writing metadata to LMDB at {LMDB_META_OUTPUT_PATH}...")
    write_metadata_to_lmdb_binary_format(
        sorted_dataset,
        LMDB_META_OUTPUT_PATH,
        LMDB_MAP_SIZE
    )
    print("Metadata written to LMDB successfully.")


if __name__ == "__main__":
    main() 