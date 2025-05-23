import cupy as cp
import numpy as np
import time
from typing import Tuple, List, Optional
from cuvs.neighbors import ivf_pq
from tqdm import tqdm

def compute_exact_knn(queries: np.ndarray, 
                     database: np.ndarray, 
                     k: int,
                     batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact k-nearest neighbors using batched processing.
    Ensures database is loaded to GPU only once.
    Assumes queries and database are np.float16.
    
    Parameters:
    -----------
    queries : np.ndarray
        Query vectors (n_queries, dim)
    database : np.ndarray
        Database vectors (n_database, dim)
    k : int
        Number of nearest neighbors to find
    batch_size : int
        Size of batches for processing queries
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        distances and indices of exact nearest neighbors
    """
    n_queries = queries.shape[0]
    n_batches = (n_queries + batch_size - 1) // batch_size
    
    # Move database to GPU ONCE, outside the loop, and ensure float16
    # print(f"Moving database ({database.shape}, {database.dtype}) to GPU as float16...") # For debugging
    database_gpu = cp.asarray(database, dtype=cp.float16)
    # print("Database moved to GPU.")

    all_distances_gpu_list = []
    all_indices_gpu_list = []
    
    for i in tqdm(range(n_batches), desc="Computing exact kNN"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_queries)
        
        query_batch_np = queries[start_idx:end_idx]
        query_batch_gpu = cp.asarray(query_batch_np, dtype=cp.float16)
        
        # Compute similarities (dot product for inner product similarity)
        # If query_batch_gpu and database_gpu are float16, similarities will be float32 by default
        similarities = cp.dot(query_batch_gpu, database_gpu.T)
        
        # Get indices of top k (unsorted)
        top_k_unsorted_idx = cp.argpartition(similarities, -k, axis=1)[:, -k:]
        # Gather the top k values (unsorted)
        top_k_unsorted_vals = cp.take_along_axis(similarities, top_k_unsorted_idx, axis=1)
        # Now, sort these top k values (descending) and get their indices
        sorted_order = cp.argsort(top_k_unsorted_vals, axis=1)[:, ::-1]
        top_k_indices_gpu = cp.take_along_axis(top_k_unsorted_idx, sorted_order, axis=1)
        top_k_similarities_gpu = cp.take_along_axis(top_k_unsorted_vals, sorted_order, axis=1)

        all_distances_gpu_list.append(top_k_similarities_gpu)
        all_indices_gpu_list.append(top_k_indices_gpu)

        # Clear memory for intermediate arrays in this batch
        del query_batch_gpu, similarities, top_k_unsorted_idx, top_k_unsorted_vals, sorted_order
        cp.get_default_memory_pool().free_all_blocks()
    
    # Free the database from GPU memory after all batches are processed
    del database_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Concatenate results on GPU
    if not all_distances_gpu_list: # Handle case with zero queries
        return np.array([]).reshape(0,k), np.array([]).reshape(0,k)

    final_distances_gpu = cp.concatenate(all_distances_gpu_list, axis=0)
    final_indices_gpu = cp.concatenate(all_indices_gpu_list, axis=0)

    # Move final results to CPU
    final_distances_cpu = cp.asnumpy(final_distances_gpu)
    final_indices_cpu = cp.asnumpy(final_indices_gpu)

    # Clear final GPU arrays
    del final_distances_gpu, final_indices_gpu, all_distances_gpu_list, all_indices_gpu_list
    cp.get_default_memory_pool().free_all_blocks()

    return final_distances_cpu, final_indices_cpu

def validate_index(index: ivf_pq.Index,
                  test_queries: np.ndarray,
                  ground_truth_indices: np.ndarray,
                  k: int = 10,
                  n_probes: int = 20,
                  n_runs: int = 5) -> Tuple[float, float, List[float]]:
    """
    Validate index quality using test queries and ground truth.
    
    Parameters:
    -----------
    index : ivf_pq.Index
        Trained IVF-PQ index
    test_queries : np.ndarray
        Test query vectors
    ground_truth_indices : np.ndarray
        Ground truth neighbor indices for test queries
    k : int
        Number of neighbors to retrieve
    n_probes : int
        Number of IVF lists to probe
    n_runs : int
        Number of search runs for timing
        
    Returns:
    --------
    Tuple[float, float, List[float]]
        recall@k, average query time (ms), list of individual recall values
    """
    # Convert queries to GPU
    # Assuming test_queries are float16 as they come from dataset_vectors
    queries_gpu = cp.asarray(test_queries, dtype=cp.float16) 
    
    # Setup search parameters
    search_params = ivf_pq.SearchParams(n_probes=n_probes)
    
    # Measure query time
    query_times = []
    # Perform one search run to get neighbors for recall calculation
    # The problem description for ivf_pq.search implies it takes queries and k.
    # Assuming it handles dtype conversion or works with float16 queries if index is compatible.
    distances_from_search, neighbors_from_search = ivf_pq.search(search_params, index, queries_gpu, k)
    cp.cuda.Stream.null.synchronize() # Ensure search is complete before timing next run / using results

    # For timing, run multiple times
    for _ in range(n_runs):
        start_time = time.time()
        # Re-run search for timing; results are not used from these runs
        _, _ = ivf_pq.search(search_params, index, queries_gpu, k)
        cp.cuda.Stream.null.synchronize()
        query_times.append((time.time() - start_time) * 1000)  # Convert to ms
    
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    
    # Calculate recall using neighbors from the first search run
    neighbors_cpu = cp.asnumpy(neighbors_from_search)
    recalls = []
    
    # Ensure ground_truth_indices has the same number of queries as neighbors_cpu
    num_queries_for_recall = min(len(neighbors_cpu), len(ground_truth_indices))

    for i in range(num_queries_for_recall):
        # gt_set = set(ground_truth_indices[i][:k]) # Ensure GT is also for top k
        # Ensure ground_truth_indices for the i-th query is also sliced to k elements
        # if ground_truth_indices itself is (n_queries, k_gt) where k_gt might be different or same as k
        gt_set = set(ground_truth_indices[i]) 
        found_set = set(neighbors_cpu[i])
        
        # Recall definition: |(retrieved relevant) INTERSECTION (all relevant ground truth)| / |all relevant ground truth|
        # Here, all relevant ground truth is the gt_set for k neighbors.
        # Retrieved relevant is the found_set.
        if not gt_set: # Avoid division by zero if a query somehow has no ground truth neighbors for k
            recalls.append(1.0 if not found_set else 0.0) # Or handle as appropriate
            continue
        recall = len(gt_set.intersection(found_set)) / len(gt_set)
        recalls.append(recall)
    
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    del queries_gpu, distances_from_search, neighbors_from_search, neighbors_cpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return avg_recall, avg_query_time, recalls

def print_validation_results(recall: float, 
                           query_time: float, 
                           n_runs_for_time: int, # Added n_runs parameter
                           individual_recalls: Optional[List[float]] = None,
                           k_for_recall: Optional[int] = None): # Added k for context
    """
    Print formatted validation results.
    """
    k_str = f"@{k_for_recall}" if k_for_recall is not None else ""
    print("\n=== Validation Results ===")
    print(f"Average Recall{k_str}: {recall:.4f}")
    print(f"Average Query Time: {query_time:.2f}ms (avg over {n_runs_for_time} runs)")
    
    if individual_recalls:
        recalls_np = np.array(individual_recalls)
        if recalls_np.size > 0:
            # k_stat_str = f" (for k={k_for_recall if k_for_recall is not None else len(individual_recalls[0]) if individual_recalls and recalls_np.ndim > 1 and recalls_np.shape[1] > 0 else 'N/A'})"
            # The k for recall stats is the same k used for avg_recall
            k_stat_str = k_str
            print(f"Recall Statistics{k_stat_str}:")
            print(f"  Min: {recalls_np.min():.4f}")
            print(f"  Max: {recalls_np.max():.4f}")
            print(f"  Std: {recalls_np.std():.4f}")
        else:
            print("Recall Statistics: No recall data to display.")
    
    print("========================\n") 