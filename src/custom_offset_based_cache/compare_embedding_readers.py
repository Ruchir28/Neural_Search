import numpy as np
import time
import logging
from pathlib import Path
import sys

# Import both approaches
from custom_offset_cache.offset_embedding_readers import OffsetEmbeddingReader, get_vectors_offset_based
from test_embedding_read import get_vectors_optimized, create_sorted_vector_selection

# Configuration
EMBEDDINGS_PATH = "lists_info/embeddings_listwise.memmap"
IVF_PQ_LAYOUT_PATH = "lists_info/layout_sorted.npz"
ID_TO_SORTED_ROW_PATH = "lists_info/vec_id_to_sorted_row.bin.npy"
VECTOR_DIM = 768

# Test parameters
NUM_VECTORS_TO_FETCH = 1000
MAX_LISTS = 40
NUM_RUNS = 5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_comparison.log'),
        logging.StreamHandler()
    ]
)

def load_common_data():
    """Load data files needed by both approaches."""
    logging.info("Loading common data files...")
    
    # Load ID to sorted row mapping
    if not Path(ID_TO_SORTED_ROW_PATH).exists():
        raise FileNotFoundError(f"ID to sorted row file not found: {ID_TO_SORTED_ROW_PATH}")
    id_to_sorted_row = np.load(ID_TO_SORTED_ROW_PATH, mmap_mode="r")
    total_embeddings = len(id_to_sorted_row)
    
    # Load layout information
    if not Path(IVF_PQ_LAYOUT_PATH).exists():
        raise FileNotFoundError(f"Layout file not found: {IVF_PQ_LAYOUT_PATH}")
    layout_data = np.load(IVF_PQ_LAYOUT_PATH)
    layout_prefix = layout_data["prefix"]
    layout_sizes = layout_data["sizes"]
    
    logging.info(f"Loaded common data: {total_embeddings:,} embeddings, {len(layout_sizes)} lists")
    
    return id_to_sorted_row, layout_prefix, layout_sizes, total_embeddings

def setup_memmap_reader(total_embeddings):
    """Setup the memmap-based reader."""
    logging.info("Setting up memmap reader...")
    
    if not Path(EMBEDDINGS_PATH).exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    
    embeddings = np.memmap(EMBEDDINGS_PATH, dtype=np.float16, mode="r", 
                          shape=(total_embeddings, VECTOR_DIM))
    
    logging.info("Memmap reader ready")
    return embeddings

def test_memmap_approach(selected_vector_ids, embeddings, id_to_sorted_row, layout_prefix, num_runs):
    """Test the memmap-based approach."""
    logging.info(f"Testing memmap approach with {len(selected_vector_ids)} vectors...")
    
    fetch_times = []
    
    for run in range(num_runs):
        fetch_start = time.perf_counter()
        
        # Use the original get_vectors function
        fetched_vectors = get_vectors_optimized(
            selected_vector_ids, embeddings, id_to_sorted_row, layout_prefix
        )
        
        fetch_end = time.perf_counter()
        fetch_time = fetch_end - fetch_start
        fetch_times.append(fetch_time)
        
        logging.info(f"Memmap run {run + 1}/{num_runs}: {fetch_time:.4f} seconds")
        
        # Verify shape on first run
        if run == 0:
            expected_shape = (len(selected_vector_ids), VECTOR_DIM)
            if fetched_vectors.shape == expected_shape:
                logging.info(f"✓ Memmap vectors shape: {fetched_vectors.shape}")
            else:
                logging.error(f"✗ Memmap shape mismatch: expected {expected_shape}, got {fetched_vectors.shape}")
    
    return fetch_times, fetched_vectors

def test_offset_approach(selected_vector_ids, id_to_sorted_row, layout_prefix, num_runs):
    """Test the offset-based approach."""
    logging.info(f"Testing offset approach with {len(selected_vector_ids)} vectors...")
    
    fetch_times = []
    fetched_vectors = None
    
    with OffsetEmbeddingReader() as reader:
        # Get read statistics
        sorted_row_indices = id_to_sorted_row[selected_vector_ids]
        stats = reader.get_read_stats(sorted_row_indices)
        logging.info(f"Offset read stats: {stats}")
        
        for run in range(num_runs):
            fetch_start = time.perf_counter()
            
            # Use the offset-based function
            fetched_vectors = get_vectors_offset_based(
                selected_vector_ids, id_to_sorted_row, layout_prefix, reader
            )
            
            fetch_end = time.perf_counter()
            fetch_time = fetch_end - fetch_start
            fetch_times.append(fetch_time)
            
            logging.info(f"Offset run {run + 1}/{num_runs}: {fetch_time:.4f} seconds")
            
            # Verify shape on first run
            if run == 0:
                expected_shape = (len(selected_vector_ids), VECTOR_DIM)
                if fetched_vectors.shape == expected_shape:
                    logging.info(f"✓ Offset vectors shape: {fetched_vectors.shape}")
                else:
                    logging.error(f"✗ Offset shape mismatch: expected {expected_shape}, got {fetched_vectors.shape}")
    
    return fetch_times, fetched_vectors, stats

def verify_results_match(memmap_vectors, offset_vectors, tolerance=1e-6):
    """Verify that both approaches return the same results."""
    logging.info("Verifying that both approaches return identical results...")
    
    if memmap_vectors.shape != offset_vectors.shape:
        logging.error(f"Shape mismatch: memmap {memmap_vectors.shape} vs offset {offset_vectors.shape}")
        return False
    
    # Check if arrays are exactly equal
    if np.array_equal(memmap_vectors, offset_vectors):
        logging.info("✓ Results are exactly identical")
        return True
    
    # Check if they're close (for floating point precision issues)
    max_diff = np.max(np.abs(memmap_vectors.astype(np.float32) - offset_vectors.astype(np.float32)))
    if max_diff < tolerance:
        logging.info(f"✓ Results are nearly identical (max diff: {max_diff:.2e})")
        return True
    else:
        logging.error(f"✗ Results differ significantly (max diff: {max_diff:.2e})")
        return False

def calculate_statistics(times, approach_name):
    """Calculate and log timing statistics."""
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    logging.info(f"{approach_name} timing statistics:")
    logging.info(f"  Average: {avg_time:.4f} seconds")
    logging.info(f"  Min:     {min_time:.4f} seconds")
    logging.info(f"  Max:     {max_time:.4f} seconds")
    logging.info(f"  Std dev: {std_time:.4f} seconds")
    logging.info(f"  Runs:    {len(times)}")
    
    return {
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
        'std': std_time,
        'runs': len(times)
    }

def run_comparison():
    """Run the complete comparison between memmap and offset approaches."""
    logging.info("=" * 80)
    logging.info("EMBEDDING READER COMPARISON TEST")
    logging.info("=" * 80)
    
    test_start_time = time.perf_counter()
    
    try:
        # Load common data
        id_to_sorted_row, layout_prefix, layout_sizes, total_embeddings = load_common_data()
        
        # Create vector selection (same as in the original test)
        logging.info("Creating vector selection...")
        selected_vector_ids = create_sorted_vector_selection(
            layout_prefix, layout_sizes, id_to_sorted_row, 
            NUM_VECTORS_TO_FETCH, MAX_LISTS, random_selection=True  # Use random selection to avoid cache bias
        )
        
        # Verify the selection spans the expected number of lists
        rows = id_to_sorted_row[selected_vector_ids]
        lists = np.searchsorted(layout_prefix, rows, 'right') - 1
        unique_lists = np.unique(lists)
        logging.info(f"Selected {len(selected_vector_ids)} vectors spanning {len(unique_lists)} lists")
        
        # Test memmap approach
        logging.info("\n" + "=" * 40)
        logging.info("TESTING MEMMAP APPROACH")
        logging.info("=" * 40)
        
        embeddings = setup_memmap_reader(total_embeddings)
        memmap_times, memmap_vectors = test_memmap_approach(
            selected_vector_ids, embeddings, id_to_sorted_row, layout_prefix, NUM_RUNS
        )
        memmap_stats = calculate_statistics(memmap_times, "Memmap")
        
        # Test offset approach
        logging.info("\n" + "=" * 40)
        logging.info("TESTING OFFSET APPROACH")
        logging.info("=" * 40)
        
        offset_times, offset_vectors, read_stats = test_offset_approach(
            selected_vector_ids, id_to_sorted_row, layout_prefix, NUM_RUNS
        )
        offset_stats = calculate_statistics(offset_times, "Offset")
        
        # Verify results match
        logging.info("\n" + "=" * 40)
        logging.info("VERIFICATION")
        logging.info("=" * 40)
        
        results_match = verify_results_match(memmap_vectors, offset_vectors)
        
        # Final comparison
        logging.info("\n" + "=" * 80)
        logging.info("FINAL COMPARISON RESULTS")
        logging.info("=" * 80)
        
        logging.info(f"Test configuration:")
        logging.info(f"  Vectors fetched: {len(selected_vector_ids)}")
        logging.info(f"  Lists spanned: {len(unique_lists)}")
        logging.info(f"  Runs per approach: {NUM_RUNS}")
        logging.info(f"  Vector dimension: {VECTOR_DIM}")
        
        logging.info(f"\nMemmap approach:")
        logging.info(f"  Average time: {memmap_stats['avg']:.4f}s")
        logging.info(f"  Best time:    {memmap_stats['min']:.4f}s")
        logging.info(f"  Worst time:   {memmap_stats['max']:.4f}s")
        logging.info(f"  Variability:  {memmap_stats['std']:.4f}s")
        
        logging.info(f"\nOffset approach:")
        logging.info(f"  Average time: {offset_stats['avg']:.4f}s")
        logging.info(f"  Best time:    {offset_stats['min']:.4f}s")
        logging.info(f"  Worst time:   {offset_stats['max']:.4f}s")
        logging.info(f"  Variability:  {offset_stats['std']:.4f}s")
        
        logging.info(f"\nOffset I/O efficiency:")
        logging.info(f"  Read operations: {read_stats['num_read_operations']}")
        logging.info(f"  Total bytes read: {read_stats['total_bytes_read']:,}")
        logging.info(f"  Useful bytes: {read_stats['useful_bytes']:,}")
        logging.info(f"  Efficiency: {read_stats['efficiency']:.1%}")
        logging.info(f"  Avg chunk size: {read_stats['avg_chunk_size']:.0f} bytes")
        
        # Performance comparison
        improvement_avg = (memmap_stats['avg'] - offset_stats['avg']) / memmap_stats['avg'] * 100
        improvement_worst = (memmap_stats['max'] - offset_stats['max']) / memmap_stats['max'] * 100
        consistency_improvement = (memmap_stats['std'] - offset_stats['std']) / memmap_stats['std'] * 100
        
        logging.info(f"\nPerformance comparison:")
        if improvement_avg > 0:
            logging.info(f"  Offset is {improvement_avg:.1f}% faster on average")
        else:
            logging.info(f"  Memmap is {-improvement_avg:.1f}% faster on average")
        
        if improvement_worst > 0:
            logging.info(f"  Offset worst case is {improvement_worst:.1f}% better")
        else:
            logging.info(f"  Memmap worst case is {-improvement_worst:.1f}% better")
        
        if consistency_improvement > 0:
            logging.info(f"  Offset is {consistency_improvement:.1f}% more consistent")
        else:
            logging.info(f"  Memmap is {-consistency_improvement:.1f}% more consistent")
        
        logging.info(f"\nResults verification: {'✓ PASSED' if results_match else '✗ FAILED'}")
        
        total_test_time = time.perf_counter() - test_start_time
        logging.info(f"\nTotal test time: {total_test_time:.2f} seconds")
        logging.info("=" * 80)
        
        return {
            'memmap_stats': memmap_stats,
            'offset_stats': offset_stats,
            'read_stats': read_stats,
            'results_match': results_match,
            'vectors_tested': len(selected_vector_ids),
            'lists_spanned': len(unique_lists)
        }
        
    except Exception as e:
        logging.error(f"Test failed with exception: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Check if offset files exist
    offset_data_path = Path("lists_info/embeddings_data.bin")
    offset_offsets_path = Path("lists_info/embeddings_offsets.npy")
    
    if not offset_data_path.exists() or not offset_offsets_path.exists():
        logging.error("Offset files not found. Please run create_offset_embeddings.py first.")
        logging.error(f"Missing files:")
        if not offset_data_path.exists():
            logging.error(f"  - {offset_data_path}")
        if not offset_offsets_path.exists():
            logging.error(f"  - {offset_offsets_path}")
        sys.exit(1)
    
    results = run_comparison()
    if results and results['results_match']:
        logging.info("✓ COMPARISON COMPLETED SUCCESSFULLY")
    else:
        logging.error("✗ COMPARISON FAILED")
        sys.exit(1) 