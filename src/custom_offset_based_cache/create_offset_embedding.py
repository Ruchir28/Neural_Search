import numpy as np
import time
import logging
from pathlib import Path

# Configuration
EMBEDDINGS_PATH = "lists_info/embeddings_listwise.memmap"
OUTPUT_DATA_PATH = "lists_info/embeddings_data.bin"
OUTPUT_OFFSETS_PATH = "lists_info/embeddings_offsets.npy"
VECTOR_DIM = 768
CHUNK_SIZE = 10000  # Process embeddings in chunks to manage memory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_offset_embeddings():
    """
    Convert memmap embeddings to offset-based format:
    - embeddings_data.bin: Raw binary data (float16)
    - embeddings_offsets.npy: Array of file offsets for each embedding
    """
    logging.info("Starting conversion to offset-based format...")
    
    # Load the existing memmap to get dimensions
    if not Path(EMBEDDINGS_PATH).exists():
        raise FileNotFoundError(f"Source embeddings file not found: {EMBEDDINGS_PATH}")
    
    # First, determine the total number of embeddings
    # We'll read a small portion to get the shape
    temp_map = np.memmap(EMBEDDINGS_PATH, dtype=np.float16, mode='r')
    total_size = temp_map.size
    total_embeddings = total_size // VECTOR_DIM
    
    logging.info(f"Source file: {EMBEDDINGS_PATH}")
    logging.info(f"Total embeddings: {total_embeddings}")
    logging.info(f"Vector dimension: {VECTOR_DIM}")
    logging.info(f"Data type: float16")
    logging.info(f"Total size: {total_size * 2 / (1024**3):.2f} GB")
    
    # Create the memmap with correct shape
    source_embeddings = np.memmap(EMBEDDINGS_PATH, dtype=np.float16, mode='r', 
                                 shape=(total_embeddings, VECTOR_DIM))
    
    # Prepare output files
    output_data_path = Path(OUTPUT_DATA_PATH)
    output_offsets_path = Path(OUTPUT_OFFSETS_PATH)
    
    # Remove existing files if they exist
    if output_data_path.exists():
        output_data_path.unlink()
        logging.info(f"Removed existing {output_data_path}")
    
    if output_offsets_path.exists():
        output_offsets_path.unlink()
        logging.info(f"Removed existing {output_offsets_path}")
    
    # Create offset array
    offsets = np.zeros(total_embeddings, dtype=np.uint64)
    
    # Process embeddings in chunks and write to binary file
    logging.info("Writing embeddings data and calculating offsets...")
    
    start_time = time.perf_counter()
    current_offset = 0
    bytes_per_embedding = VECTOR_DIM * 2  # float16 = 2 bytes per element
    
    with open(output_data_path, 'wb') as data_file:
        for chunk_start in range(0, total_embeddings, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_embeddings)
            chunk_size = chunk_end - chunk_start
            
            # Read chunk from source
            chunk_data = source_embeddings[chunk_start:chunk_end]
            
            # Record offsets for this chunk
            for i in range(chunk_size):
                offsets[chunk_start + i] = current_offset
                current_offset += bytes_per_embedding
            
            # Write chunk to output file
            chunk_data.tobytes()  # Convert to bytes
            data_file.write(chunk_data.tobytes())
            
            # Progress logging
            if (chunk_start // CHUNK_SIZE) % 100 == 0:
                progress = (chunk_end / total_embeddings) * 100
                elapsed = time.perf_counter() - start_time
                logging.info(f"Progress: {progress:.1f}% ({chunk_end}/{total_embeddings}) - "
                           f"Elapsed: {elapsed:.1f}s")
    
    # Save offsets array
    logging.info("Saving offsets array...")
    np.save(output_offsets_path, offsets)
    
    # Verify the output
    data_file_size = output_data_path.stat().st_size
    expected_size = total_embeddings * bytes_per_embedding
    
    logging.info("Verification:")
    logging.info(f"Data file size: {data_file_size:,} bytes ({data_file_size / (1024**3):.2f} GB)")
    logging.info(f"Expected size: {expected_size:,} bytes ({expected_size / (1024**3):.2f} GB)")
    logging.info(f"Offsets array shape: {offsets.shape}")
    logging.info(f"Offsets file size: {output_offsets_path.stat().st_size:,} bytes")
    
    if data_file_size == expected_size:
        logging.info("✓ Data file size matches expected size")
    else:
        logging.error("✗ Data file size mismatch!")
        return False
    
    # Test reading a few embeddings to verify format
    logging.info("Testing random access...")
    test_indices = [0, total_embeddings // 2, total_embeddings - 1]
    
    with open(output_data_path, 'rb') as data_file:
        for test_idx in test_indices:
            # Read using offset
            offset = offsets[test_idx]
            data_file.seek(offset)
            raw_data = data_file.read(bytes_per_embedding)
            reconstructed = np.frombuffer(raw_data, dtype=np.float16).reshape(VECTOR_DIM)
            
            # Compare with original
            original = source_embeddings[test_idx]
            
            if np.array_equal(reconstructed, original):
                logging.info(f"✓ Test {test_idx}: Data matches original")
            else:
                logging.error(f"✗ Test {test_idx}: Data mismatch!")
                return False
    
    total_time = time.perf_counter() - start_time
    logging.info(f"Conversion completed successfully in {total_time:.2f} seconds")
    logging.info(f"Output files:")
    logging.info(f"  Data: {output_data_path}")
    logging.info(f"  Offsets: {output_offsets_path}")
    
    return True

if __name__ == "__main__":
    try:
        success = create_offset_embeddings()
        if success:
            logging.info("✓ CONVERSION COMPLETED SUCCESSFULLY")
        else:
            logging.error("✗ CONVERSION FAILED")
    except Exception as e:
        logging.error(f"Conversion failed with exception: {e}", exc_info=True) 