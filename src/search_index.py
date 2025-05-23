import cupy as cp
import numpy as np
import time
from pathlib import Path
from cuvs.neighbors import ivf_pq
import os # For os.path.exists if needed, though Path.exists is preferred
import sys # For redirecting stdout
import datetime # For timestamped log files

# Import necessary functions from train_index and utils
from train_index import prepare_training_data, N_TRAIN, VECTOR_DIM, DATASET, CACHE_DIR 
from utils.gpu_memory_monitor import print_memory_stats
from utils.validation import compute_exact_knn, validate_index, print_validation_results

# Configuration (should ideally match the training configuration)
# These are imported from train_index:
# DATASET = "Cohere/wikipedia-22-12-en-embeddings"
# CACHE_DIR = "dataset_cache"
# N_TRAIN = 10000  # Number of vectors to load for validation/test query generation
# VECTOR_DIM = 768 # Dimension of embeddings

# Parameters for validation
N_TEST_QUERIES = 1000
K_NEIGHBORS = 10
N_PROBES_SEARCH = 40 # Example value, can be tuned

def main_search():
    # --- Setup Logging --- 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("search_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"search_results_{timestamp}.log"
    
    original_stdout = sys.stdout  # Save a reference to the original standard output
    print(f"Starting search. Output will be logged to: {log_file_path.resolve()}") # Initial message to console

    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file  # Redirect stdout to log_file
        
        try:
            # Define the path to the trained index
            output_dir = Path("trained_indices")
            trained_index_path = output_dir / "trained_ivfpq_index_full.bin"

            if not trained_index_path.exists():
                print(f"Error: Trained index not found at {trained_index_path}")
                print(f"Please run src/train_index.py first to train and save an index, and then src/add_embeddings_to_index.py to populate it fully.")
                return

            print(f"\nLoading pre-trained index from {trained_index_path}...")
            load_start_time = time.time()
            trained_index = ivf_pq.load(str(trained_index_path))
            load_time = time.time() - load_start_time
            print(f"Index loaded successfully in {load_time:.2f} seconds.")
            print_memory_stats()

            # Load dataset vectors (needed for test queries and ground truth)
            print(f"\nLoading dataset vectors (up to {N_TRAIN}) for validation query generation and ground truth...")
            data_prep_start_time = time.time()
            dataset_vectors = prepare_training_data(n_train=N_TRAIN) 
            data_prep_time = time.time() - data_prep_start_time
            print(f"Dataset vectors loaded in {data_prep_time:.2f} seconds.")

            if len(dataset_vectors) == 0:
                print("Error: No dataset vectors loaded. Cannot proceed with validation.")
                return
            
            actual_n_test_queries = min(N_TEST_QUERIES, len(dataset_vectors))
            if actual_n_test_queries == 0:
                print("Error: No vectors available for test queries after adjustment. Skipping validation.")
                return
            if N_TEST_QUERIES > len(dataset_vectors):
                print(f"Warning: Requested N_TEST_QUERIES ({N_TEST_QUERIES}) > available dataset vectors ({len(dataset_vectors)}). Using {actual_n_test_queries}.")

            print(f"\nSelecting {actual_n_test_queries} random test queries...")
            test_indices = np.random.choice(len(dataset_vectors), size=actual_n_test_queries, replace=False)
            test_queries = dataset_vectors[test_indices]

            print(f"\nComputing ground truth for {actual_n_test_queries} queries (k={K_NEIGHBORS})... (against N_TRAIN={N_TRAIN} subset)")
            gt_compute_start_time = time.time()
            # Reduce batch_size for compute_exact_knn to manage memory during argsort
            knn_batch_size = 10 # Was 100, reduced to 10
            gt_distances, gt_indices = compute_exact_knn(
                test_queries, dataset_vectors, k=K_NEIGHBORS, batch_size=knn_batch_size
            )
            gt_compute_time = time.time() - gt_compute_start_time
            print(f"Ground truth computation completed in {gt_compute_time:.2f} seconds.")

            print(f"\nValidating index (k={K_NEIGHBORS}, n_probes={N_PROBES_SEARCH})...")
            val_start_time = time.time()
            # Assuming n_runs for validate_index is 5 by default or a known value
            num_runs_for_timing = validate_index.__defaults__[2] if validate_index.__defaults__ and len(validate_index.__defaults__) > 2 else 5 # Get n_runs from default or assume 5
            recall, avg_query_time_ms, recalls_list = validate_index(
                trained_index, test_queries, gt_indices, k=K_NEIGHBORS, n_probes=N_PROBES_SEARCH, n_runs=num_runs_for_timing
            )
            val_time = time.time() - val_start_time
            
            # print_validation_results(recall, avg_query_time_ms, recalls_list) # OLD CALL
            print_validation_results(recall, avg_query_time_ms, n_runs_for_time=num_runs_for_timing, individual_recalls=recalls_list, k_for_recall=K_NEIGHBORS) # NEW CALL
            print(f"Overall validation process (search part for {num_runs_for_timing} runs) took: {val_time:.2f} seconds.")

            # --- New: Oversample + rerank test ---
            OVERSAMPLE_K = 1000
            FINAL_K = K_NEIGHBORS
            print(f"\nRunning IVF-PQ oversample (top {OVERSAMPLE_K}) + rerank (dot product) test...")
            rerank_start_time = time.time()
            queries_gpu = cp.asarray(test_queries)
            search_params = ivf_pq.SearchParams(n_probes=N_PROBES_SEARCH)
            distances_pq, candidates_pq = ivf_pq.search(search_params, trained_index, queries_gpu, OVERSAMPLE_K)
            candidates_cpu = cp.asnumpy(candidates_pq)
            reranked_neighbors = np.zeros((actual_n_test_queries, FINAL_K), dtype=np.int64)
            # reranked_scores = np.zeros((actual_n_test_queries, FINAL_K), dtype=np.float32) # Not used currently
            for i in range(actual_n_test_queries):
                query = test_queries[i]
                candidate_indices = candidates_cpu[i]
                # Ensure candidate indices are valid and within bounds of dataset_vectors for reranking
                # This part is tricky because candidates_cpu are indices from the *full* index (35M)
                # but dataset_vectors for reranking is only the N_TRAIN subset (2M)
                # For this reranking to be meaningful with current dataset_vectors, candidates must be < N_TRAIN
                
                # Let's fetch original vectors for candidates that are within the N_TRAIN range
                # Other candidates from the larger index cannot be reranked with dataset_vectors
                valid_candidate_indices_for_rerank = candidate_indices[candidate_indices < len(dataset_vectors)]
                
                if len(valid_candidate_indices_for_rerank) == 0:
                    # No candidates from the PQ search fall within our reranking dataset_vectors subset
                    # This query will have 0 recall for the reranked part, or we fill with dummy values
                    reranked_neighbors[i] = -1 # Or some other indicator
                    continue

                candidate_vecs = dataset_vectors[valid_candidate_indices_for_rerank]
                scores = np.dot(candidate_vecs, query)
                
                # We need to get top FINAL_K from the *scored* candidates
                # and their original indices from candidates_cpu
                num_rerank_candidates = len(valid_candidate_indices_for_rerank)
                actual_final_k = min(FINAL_K, num_rerank_candidates)

                if actual_final_k > 0:
                    topk_rerank_idx = np.argsort(scores)[-actual_final_k:][::-1]
                    reranked_neighbors[i, :actual_final_k] = valid_candidate_indices_for_rerank[topk_rerank_idx]
                    if actual_final_k < FINAL_K:
                        reranked_neighbors[i, actual_final_k:] = -1 # Pad if fewer than FINAL_K results
                else:
                     reranked_neighbors[i] = -1 # Or fill with dummy values
            
            recalls = []
            for i in range(actual_n_test_queries):
                if np.all(reranked_neighbors[i] == -1):
                    recalls.append(0.0)
                    continue
                gt_set = set(gt_indices[i])
                # Filter out -1 padding if any before creating found_set
                found_set = set(val for val in reranked_neighbors[i] if val != -1)
                if len(gt_set) == 0: # Should not happen with K_NEIGHBORS > 0
                    recalls.append(0.0 if len(found_set) > 0 else 1.0) # Or handle as appropriate
                    continue
                recall_val = len(gt_set.intersection(found_set)) / len(gt_set) # Recall is based on gt_set size
                recalls.append(recall_val)
            
            avg_recall = np.mean(recalls)
            min_recall = np.min(recalls)
            max_recall = np.max(recalls)
            std_recall = np.std(recalls)
            rerank_time = time.time() - rerank_start_time
            print("\n=== Oversample + Rerank Validation Results ===")
            print(f"Average Recall@{FINAL_K} (reranked on N_TRAIN subset): {avg_recall:.4f}")
            print(f"Recall Min: {min_recall:.4f}  Max: {max_recall:.4f}  Std: {std_recall:.4f}")
            print(f"Total time for oversample + rerank: {rerank_time:.2f} seconds")
            print("Note: Reranking for this test is performed using the original vectors from the N_TRAIN subset only.")
            print("Candidates from the IVF-PQ search that are outside this N_TRAIN subset are not included in reranking scores.")
            print("============================================\n")

            print("\nSearch and validation pipeline completed.")
            total_script_time = load_time + data_prep_time + gt_compute_time + val_time + rerank_time
            print(f"Total script time (index load + data load + GT compute + validation search + rerank): {total_script_time:.2f} seconds")
            
            print(f"\n--- LOGGING COMPLETE --- Full output in {log_file_path.resolve()} ---")

        finally:
            sys.stdout = original_stdout # Reset stdout to original

    # Final message to console after logging is done
    print(f"Search process completed. Results also saved to: {log_file_path.resolve()}")


if __name__ == "__main__":
    from train_index import N_TRAIN, VECTOR_DIM, DATASET, CACHE_DIR, prepare_training_data
    from utils.gpu_memory_monitor import print_memory_stats
    from utils.validation import compute_exact_knn, validate_index, print_validation_results
    
    # Ensure N_TEST_QUERIES, K_NEIGHBORS, N_PROBES_SEARCH are defined if not already global
    # These are usually defined at the top of search_index.py
    # For safety, if they were defined inside main_search, ensure they are accessible or redefined here if needed
    # (Assuming they are module-level globals as per typical structure)
    if 'N_TEST_QUERIES' not in globals(): N_TEST_QUERIES = 1000
    if 'K_NEIGHBORS' not in globals(): K_NEIGHBORS = 10
    if 'N_PROBES_SEARCH' not in globals(): N_PROBES_SEARCH = 40

    main_search() 