import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Tuple
import os

# Configuration
EMBEDDINGS_DATA_PATH = "lists_info/embeddings_data.bin"
EMBEDDINGS_OFFSETS_PATH = "lists_info/embeddings_offsets.npy"
VECTOR_DIM = 768
BYTES_PER_EMBEDDING = VECTOR_DIM * 2  # float16 = 2 bytes per element

# Batching parameters for optimal I/O
CHUNK_SIZE_BYTES = 256 * 1024  # 256KB chunks for good I/O performance
MAX_GAP_BYTES = 64 * 1024      # If gap between embeddings < 64KB, read continuously

class OffsetEmbeddingReader:
    """
    Optimized embedding reader using file offsets for predictable performance.
    """
    
    def __init__(self, data_path: str = EMBEDDINGS_DATA_PATH, 
                 offsets_path: str = EMBEDDINGS_OFFSETS_PATH):
        """
        Initialize the reader with data and offset files.
        
        Args:
            data_path: Path to the binary embedding data file
            offsets_path: Path to the numpy offsets array file
        """
        self.data_path = Path(data_path)
        self.offsets_path = Path(offsets_path)
        
        # Validate files exist
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not self.offsets_path.exists():
            raise FileNotFoundError(f"Offsets file not found: {offsets_path}")
        
        # Load offsets into memory for fast lookup
        logging.info(f"Loading offsets from {offsets_path}...")
        start_time = time.perf_counter()
        self.offsets = np.load(offsets_path)  # Load fully into RAM for fastest access
        load_time = time.perf_counter() - start_time
        
        self.total_embeddings = len(self.offsets)
        logging.info(f"Loaded {self.total_embeddings:,} offsets in {load_time:.4f}s")
        
        # File handle will be opened when needed
        self._file_handle = None
    
    def __enter__(self):
        """Context manager entry."""
        self._file_handle = open(self.data_path, 'rb')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
    
    def _plan_reads(self, embedding_ids: np.ndarray) -> List[Tuple[int, int, List[int]]]:
        """
        Plan optimal read operations by grouping nearby embeddings.
        
        Args:
            embedding_ids: Array of embedding IDs to read
            
        Returns:
            List of (start_offset, end_offset, id_positions) tuples
            where id_positions maps to original embedding_ids order
        """
        # Get offsets for requested embeddings
        requested_offsets = [(self.offsets[eid], i, eid) for i, eid in enumerate(embedding_ids)]
        
        # Sort by file offset for sequential access
        requested_offsets.sort(key=lambda x: x[0])
        
        # Group nearby offsets into chunks
        chunks = []
        current_chunk_start = None
        current_chunk_end = None
        current_chunk_ids = []
        
        for offset, original_pos, embedding_id in requested_offsets:
            embedding_end = offset + BYTES_PER_EMBEDDING
            
            if current_chunk_start is None:
                # Start new chunk
                current_chunk_start = offset
                current_chunk_end = embedding_end
                current_chunk_ids = [(original_pos, embedding_id, offset)]
            else:
                # Check if we should extend current chunk or start new one
                gap = offset - current_chunk_end
                chunk_size_if_extended = embedding_end - current_chunk_start
                
                if gap <= MAX_GAP_BYTES and chunk_size_if_extended <= CHUNK_SIZE_BYTES:
                    # Extend current chunk
                    current_chunk_end = embedding_end
                    current_chunk_ids.append((original_pos, embedding_id, offset))
                else:
                    # Finalize current chunk and start new one
                    chunks.append((current_chunk_start, current_chunk_end, current_chunk_ids))
                    current_chunk_start = offset
                    current_chunk_end = embedding_end
                    current_chunk_ids = [(original_pos, embedding_id, offset)]
        
        # Don't forget the last chunk
        if current_chunk_start is not None:
            chunks.append((current_chunk_start, current_chunk_end, current_chunk_ids))
        
        return chunks
    
    def get_vectors(self, embedding_ids: np.ndarray) -> np.ndarray:
        """
        Fetch embeddings for given IDs using optimized offset-based reading.
        
        Args:
            embedding_ids: Array of embedding IDs to fetch
            
        Returns:
            Array of shape (len(embedding_ids), VECTOR_DIM) with embeddings
        """
        if len(embedding_ids) == 0:
            return np.empty((0, VECTOR_DIM), dtype=np.float16)
        
        # Validate embedding IDs
        max_id = np.max(embedding_ids)
        if max_id >= self.total_embeddings:
            raise ValueError(f"Embedding ID {max_id} exceeds maximum {self.total_embeddings - 1}")
        
        # Plan optimal read operations
        read_chunks = self._plan_reads(embedding_ids)
        
        # Prepare output array
        result = np.empty((len(embedding_ids), VECTOR_DIM), dtype=np.float16)
        
        # Execute reads
        if self._file_handle is None:
            raise RuntimeError("Reader not opened. Use 'with OffsetEmbeddingReader() as reader:'")
        
        for chunk_start, chunk_end, chunk_ids in read_chunks:
            # Read the entire chunk
            chunk_size = int(chunk_end - chunk_start)  # Convert to Python int
            self._file_handle.seek(int(chunk_start))  # Convert to Python int
            chunk_data = self._file_handle.read(chunk_size)
            
            # Extract individual embeddings from the chunk
            for original_pos, embedding_id, embedding_offset in chunk_ids:
                # Calculate position within the chunk
                pos_in_chunk = int(embedding_offset - chunk_start)  # Convert to Python int
                
                # Extract embedding data
                embedding_bytes = chunk_data[pos_in_chunk:pos_in_chunk + BYTES_PER_EMBEDDING]
                embedding_vector = np.frombuffer(embedding_bytes, dtype=np.float16)
                
                # Store in result array at original position
                result[original_pos] = embedding_vector
        
        return result
    
    def get_read_stats(self, embedding_ids: np.ndarray) -> dict:
        """
        Get statistics about read operations for given embedding IDs.
        Useful for understanding I/O efficiency.
        
        Args:
            embedding_ids: Array of embedding IDs
            
        Returns:
            Dictionary with read statistics
        """
        read_chunks = self._plan_reads(embedding_ids)
        
        total_bytes_read = sum(chunk_end - chunk_start for chunk_start, chunk_end, _ in read_chunks)
        useful_bytes = len(embedding_ids) * BYTES_PER_EMBEDDING
        
        return {
            'num_embeddings': len(embedding_ids),
            'num_read_operations': len(read_chunks),
            'total_bytes_read': total_bytes_read,
            'useful_bytes': useful_bytes,
            'efficiency': useful_bytes / total_bytes_read if total_bytes_read > 0 else 0,
            'avg_chunk_size': total_bytes_read / len(read_chunks) if read_chunks else 0
        }

def get_vectors_offset_based(ids: np.ndarray, id_to_sorted_row: np.ndarray, 
                           layout_prefix: np.ndarray, reader: OffsetEmbeddingReader) -> np.ndarray:
    """
    Drop-in replacement for the original get_vectors function using offset-based reading.
    Maintains the same interface for easy integration.
    
    Args:
        ids: Original vector IDs to fetch
        id_to_sorted_row: Mapping from original IDs to sorted row indices
        layout_prefix: Layout prefix array (not used in offset approach, kept for compatibility)
        reader: OffsetEmbeddingReader instance
        
    Returns:
        Array of embeddings with shape (len(ids), VECTOR_DIM)
    """
    # Convert original IDs to sorted row indices (which are our embedding IDs in the offset file)
    sorted_row_indices = id_to_sorted_row[ids]
    
    # Use the offset reader to fetch embeddings
    return reader.get_vectors(sorted_row_indices)

# Convenience function for quick testing
def test_offset_reader():
    """Test the offset reader with some sample data."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test with a few random embedding IDs
    test_ids = np.array([0, 100, 1000, 5000, 10000])
    
    with OffsetEmbeddingReader() as reader:
        logging.info(f"Testing with embedding IDs: {test_ids}")
        
        # Get read statistics
        stats = reader.get_read_stats(test_ids)
        logging.info(f"Read stats: {stats}")
        
        # Perform the actual read
        start_time = time.perf_counter()
        embeddings = reader.get_vectors(test_ids)
        read_time = time.perf_counter() - start_time
        
        logging.info(f"Read {len(embeddings)} embeddings in {read_time:.4f}s")
        logging.info(f"Embeddings shape: {embeddings.shape}")
        logging.info(f"Data type: {embeddings.dtype}")
        
        return embeddings, read_time, stats

if __name__ == "__main__":
    test_offset_reader() 