import numpy as np
import time
import logging
from pathlib import Path
import random

# Configuration paths (matching the server configuration)
EMBEDDINGS_PATH = "lists_info/embeddings_listwise.memmap"
IVF_PQ_LAYOUT_PATH = "lists_info/layout_sorted.npz"
ID_TO_SORTED_ROW_PATH = "lists_info/vec_id_to_sorted_row.bin.npy"
VECTOR_DIM = 768

# Test parameters
NUM_VECTORS_TO_FETCH = 1000
MAX_LISTS = 40

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_embedding_read.log'),
        logging.StreamHandler()
    ]
)

def get_vectors_optimized(ids: np.ndarray, emb_map: np.memmap, id_to_sorted_row_map: np.ndarray, current_layout_prefix: np.ndarray) -> np.ndarray:
    """
    Optimized version of get_vectors function from server.py
    Fetches full embedding vectors for given original vector IDs.
    """
    rows = id_to_sorted_row_map[ids]  # Convert original IDs to sorted row indices
    # Determine which list each sorted row index belongs to
    lists = np.searchsorted(current_layout_prefix, rows, 'right') - 1
    out = np.empty((len(ids), VECTOR_DIM), dtype=np.float16)

    for list_idx in np.unique(lists):
        start_offset = current_layout_prefix[list_idx]
        end_offset = current_layout_prefix[list_idx + 1] 

        mask = (lists == list_idx)
        # Offsets within the specific list's block in emb_map
        offsets_in_list_block = rows[mask] - start_offset
        out[mask] = emb_map[start_offset:end_offset][offsets_in_list_block]

    return out

def create_sorted_vector_selection(layout_prefix: np.ndarray, layout_sizes: np.ndarray, 
                                 id_to_sorted_row: np.ndarray, num_vectors: int, max_lists: int, random_selection: bool = True):
    """
    Creates a sorted selection of vector IDs that span at most max_lists lists.
    Returns vector IDs in sorted order by their row indices.
    
    Args:
        random_selection: If True, randomly select lists. If False, select largest lists.
    """
    logging.info(f"Creating sorted selection of {num_vectors} vectors from at most {max_lists} lists")
    
    # Find lists with sufficient vectors
    non_empty_lists = np.where(layout_sizes > 0)[0]
    logging.info(f"Found {len(non_empty_lists)} non-empty lists")
    
    # Select lists that can contribute to our target
    if random_selection:
        # Randomly select lists to avoid page cache bias
        selected_lists = random.sample(list(non_empty_lists), min(max_lists, len(non_empty_lists)))
        selected_lists.sort()  # Sort for consistent ordering
        logging.info(f"Randomly selected {len(selected_lists)} lists: {selected_lists[:10]}{'...' if len(selected_lists) > 10 else ''}")
    else:
        # Prioritize larger lists for better distribution (original behavior)
        list_priorities = [(list_idx, layout_sizes[list_idx]) for list_idx in non_empty_lists]
        list_priorities.sort(key=lambda x: x[1], reverse=True)  # Sort by size descending
        selected_lists = [list_idx for list_idx, _ in list_priorities[:max_lists]]
        selected_lists.sort()  # Sort list indices for consistent ordering
        logging.info(f"Selected {len(selected_lists)} largest lists: {selected_lists[:10]}{'...' if len(selected_lists) > 10 else ''}")
    
    # Calculate how many vectors to take from each list
    total_available = sum(layout_sizes[list_idx] for list_idx in selected_lists)
    logging.info(f"Total available vectors in selected lists: {total_available}")
    
    if total_available < num_vectors:
        logging.warning(f"Only {total_available} vectors available, less than requested {num_vectors}")
        num_vectors = total_available
    
    # Distribute vectors proportionally across selected lists
    vectors_per_list = []
    remaining_vectors = num_vectors
    
    for i, list_idx in enumerate(selected_lists):
        if i == len(selected_lists) - 1:  # Last list gets remaining vectors
            vectors_from_this_list = remaining_vectors
        else:
            # Proportional allocation
            proportion = layout_sizes[list_idx] / total_available
            vectors_from_this_list = min(int(proportion * num_vectors), 
                                       layout_sizes[list_idx], 
                                       remaining_vectors)
        
        vectors_per_list.append(vectors_from_this_list)
        remaining_vectors -= vectors_from_this_list
        
        if remaining_vectors <= 0:
            break
    
    logging.info(f"Vectors per list: {dict(zip(selected_lists, vectors_per_list))}")
    
    # Now select actual vector IDs
    selected_vector_ids = []
    
    for list_idx, num_from_list in zip(selected_lists, vectors_per_list):
        if num_from_list <= 0:
            continue
            
        # Get the range of sorted rows for this list
        start_row = layout_prefix[list_idx]
        end_row = layout_prefix[list_idx + 1]
        
        # Find original vector IDs that map to this row range
        # This is the inverse operation of id_to_sorted_row
        candidate_ids = []
        for original_id in range(len(id_to_sorted_row)):
            sorted_row = id_to_sorted_row[original_id]
            if start_row <= sorted_row < end_row:
                candidate_ids.append(original_id)
                if len(candidate_ids) >= num_from_list:
                    break
        
        # Take the first num_from_list IDs (they're already in sorted order by construction)
        selected_from_this_list = candidate_ids[:num_from_list]
        selected_vector_ids.extend(selected_from_this_list)
        
        logging.info(f"List {list_idx}: selected {len(selected_from_this_list)} vectors "
                    f"(rows {start_row}-{end_row-1})")
    
    # Sort the final selection by their sorted row indices for optimal access
    selected_vector_ids = np.array(selected_vector_ids)
    sorted_rows = id_to_sorted_row[selected_vector_ids]
    sort_order = np.argsort(sorted_rows)
    selected_vector_ids = selected_vector_ids[sort_order]
    
    logging.info(f"Final selection: {len(selected_vector_ids)} vectors in sorted order")
    
    return selected_vector_ids

def verify_list_distribution(vector_ids: np.ndarray, id_to_sorted_row: np.ndarray, 
                           layout_prefix: np.ndarray, max_lists: int):
    """
    Verify that the selected vectors come from at most max_lists lists.
    """
    rows = id_to_sorted_row[vector_ids]
    lists = np.searchsorted(layout_prefix, rows, 'right') - 1
    unique_lists = np.unique(lists)
    
    logging.info(f"Verification: {len(vector_ids)} vectors span {len(unique_lists)} lists")
    logging.info(f"Lists used: {unique_lists}")
    
    if len(unique_lists) <= max_lists:
        logging.info(f"✓ PASS: Using {len(unique_lists)} lists (≤ {max_lists})")
        return True
    else:
        logging.error(f"✗ FAIL: Using {len(unique_lists)} lists (> {max_lists})")
        return False

def run_vector_fetch_test():
    """
    Main test function that fetches 1000 vectors from memmap with timing.
    """
    logging.info("=" * 60)
    logging.info("STARTING VECTOR FETCH TEST")
    logging.info("=" * 60)
    
    test_start_time = time.perf_counter()
    
    # Load required data files
    logging.info("Loading data files...")
    load_start = time.perf_counter()
    
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
    
    # Load embeddings memmap
    if not Path(EMBEDDINGS_PATH).exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    embeddings = np.memmap(EMBEDDINGS_PATH, dtype=np.float16, mode="r", 
                          shape=(total_embeddings, VECTOR_DIM))
    
    load_time = time.perf_counter() - load_start
    logging.info(f"Data loading completed in {load_time:.4f} seconds")
    logging.info(f"Total embeddings: {total_embeddings}")
    logging.info(f"Number of lists: {len(layout_sizes)}")
    logging.info(f"Non-empty lists: {np.sum(layout_sizes > 0)}")
    
    # Create sorted vector selection
    selection_start = time.perf_counter()
    selected_vector_ids = create_sorted_vector_selection(
        layout_prefix, layout_sizes, id_to_sorted_row, 
        NUM_VECTORS_TO_FETCH, MAX_LISTS
    )
    selection_time = time.perf_counter() - selection_start
    logging.info(f"Vector selection completed in {selection_time:.4f} seconds")
    
    # Verify list distribution
    verify_start = time.perf_counter()
    verification_passed = verify_list_distribution(
        selected_vector_ids, id_to_sorted_row, layout_prefix, MAX_LISTS
    )
    verify_time = time.perf_counter() - verify_start
    logging.info(f"Verification completed in {verify_time:.4f} seconds")
    
    if not verification_passed:
        logging.error("Test failed verification - aborting")
        return
    
    # Perform the actual vector fetch with timing
    logging.info(f"Fetching {len(selected_vector_ids)} vectors from memmap...")
    
    # Multiple runs for accurate timing
    num_runs = 5
    fetch_times = []
    
    for run in range(num_runs):
        fetch_start = time.perf_counter()
        
        # This is the main operation we're timing
        fetched_vectors = get_vectors_optimized(
            selected_vector_ids, embeddings, id_to_sorted_row, layout_prefix
        )
        
        fetch_end = time.perf_counter()
        fetch_time = fetch_end - fetch_start
        fetch_times.append(fetch_time)
        
        logging.info(f"Run {run + 1}/{num_runs}: {fetch_time:.4f} seconds")
        
        # Verify the shape of fetched vectors
        if run == 0:  # Only verify once
            expected_shape = (len(selected_vector_ids), VECTOR_DIM)
            if fetched_vectors.shape == expected_shape:
                logging.info(f"✓ Fetched vectors shape: {fetched_vectors.shape}")
            else:
                logging.error(f"✗ Shape mismatch: expected {expected_shape}, got {fetched_vectors.shape}")
    
    # Calculate timing statistics
    avg_fetch_time = np.mean(fetch_times)
    min_fetch_time = np.min(fetch_times)
    max_fetch_time = np.max(fetch_times)
    std_fetch_time = np.std(fetch_times)
    
    total_test_time = time.perf_counter() - test_start_time
    
    # Results summary
    logging.info("=" * 60)
    logging.info("TEST RESULTS SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Vectors fetched: {len(selected_vector_ids)}")
    logging.info(f"Lists used: {len(np.unique(np.searchsorted(layout_prefix, id_to_sorted_row[selected_vector_ids], 'right') - 1))}")
    logging.info(f"Max lists allowed: {MAX_LISTS}")
    logging.info(f"Vector dimension: {VECTOR_DIM}")
    logging.info(f"Data type: {fetched_vectors.dtype}")
    logging.info("")
    logging.info("TIMING RESULTS:")
    logging.info(f"  Average fetch time: {avg_fetch_time:.4f} seconds")
    logging.info(f"  Min fetch time:     {min_fetch_time:.4f} seconds")
    logging.info(f"  Max fetch time:     {max_fetch_time:.4f} seconds")
    logging.info(f"  Std deviation:      {std_fetch_time:.4f} seconds")
    logging.info(f"  Runs performed:     {num_runs}")
    logging.info("")
    logging.info("BREAKDOWN:")
    logging.info(f"  Data loading:       {load_time:.4f} seconds")
    logging.info(f"  Vector selection:   {selection_time:.4f} seconds")
    logging.info(f"  Verification:       {verify_time:.4f} seconds")
    logging.info(f"  Total test time:    {total_test_time:.4f} seconds")
    logging.info("")
    logging.info(f"Throughput: {len(selected_vector_ids) / avg_fetch_time:.0f} vectors/second")
    logging.info(f"Data rate: {(len(selected_vector_ids) * VECTOR_DIM * 2) / (avg_fetch_time * 1024 * 1024):.2f} MB/second")
    logging.info("=" * 60)
    
    return {
        'vectors_fetched': len(selected_vector_ids),
        'lists_used': len(np.unique(np.searchsorted(layout_prefix, id_to_sorted_row[selected_vector_ids], 'right') - 1)),
        'avg_fetch_time': avg_fetch_time,
        'min_fetch_time': min_fetch_time,
        'max_fetch_time': max_fetch_time,
        'std_fetch_time': std_fetch_time,
        'total_test_time': total_test_time,
        'verification_passed': verification_passed
    }

if __name__ == "__main__":
    try:
        results = run_vector_fetch_test()
        if results and results['verification_passed']:
            logging.info("✓ TEST COMPLETED SUCCESSFULLY")
        else:
            logging.error("✗ TEST FAILED")
    except Exception as e:
        logging.error(f"Test failed with exception: {e}", exc_info=True)
