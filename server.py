import cupy as cp
import numpy as np
from pathlib import Path
from cuvs.neighbors import ivf_pq
import cohere
import time
import logging
import lmdb
import struct
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from config.env file
load_dotenv('config.env')

# Configurable paths via environment variables
DATA_DIR = os.getenv("DATA_DIR", ".")
INDEX_PATH = Path(os.getenv("INDEX_PATH", f"{DATA_DIR}/trained_indices/trained_ivfpq_index_full.bin"))
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", f"{DATA_DIR}/lists_info/embeddings_listwise.memmap")
IVF_PQ_LAYOUT_PATH = os.getenv("IVF_PQ_LAYOUT_PATH", f"{DATA_DIR}/lists_info/layout_sorted.npz")
ID_TO_SORTED_ROW_PATH = os.getenv("ID_TO_SORTED_ROW_PATH", f"{DATA_DIR}/lists_info/vec_id_to_sorted_row.bin.npy")
LMDB_META_PATH = os.getenv("LMDB_META_PATH", f"{DATA_DIR}/lists_info/metadata_25gb.lmdb")

# Performance configuration
LOAD_EMBEDDINGS_TO_RAM = os.getenv("LOAD_EMBEDDINGS_TO_RAM", "false").lower() in ("true", "1", "yes")

OVERSAMPLE_K = 1000  
FINAL_K = 15  
N_PROBES_SEARCH = 40  
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
VECTOR_DIM = 768  

index_global = None
cohere_client_global = None
embeddings_global = None
layout_prefix_global = None
id_to_sorted_row_global = None
lmdb_env_global = None
lmdb_metadata_db_global = None
thread_pool_global = None

logging.basicConfig(
    filename='server_query.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

app = FastAPI(title="Semantic Search Server")

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = FINAL_K

class ResultItem(BaseModel):
    rank: int
    score: float
    text: str

class QueryResponse(BaseModel):
    results: list[ResultItem]
    timings: dict


def get_vectors_server(ids: np.ndarray, emb_map: np.ndarray, id_to_sorted_row_map: np.ndarray, current_layout_prefix: np.ndarray) -> np.ndarray:
    start_time = time.perf_counter()
    rows = id_to_sorted_row_map[ids]
    lists = np.searchsorted(current_layout_prefix, rows, 'right') - 1
    out = np.empty((len(ids), VECTOR_DIM), dtype=np.float16)
    print(f"Setup time: {time.perf_counter() - start_time:.4f} s")

    def process_list(list_idx):
        start = time.perf_counter()
        start_offset = current_layout_prefix[list_idx]
        end_offset = current_layout_prefix[list_idx + 1]
        mask = (lists == list_idx)
        offsets_in_list_block = rows[mask] - start_offset
        vectors = emb_map[start_offset:end_offset][offsets_in_list_block]
        print(f"List {list_idx} fetch time: {time.perf_counter() - start:.4f} s")
        return mask, vectors

    unique_lists = np.unique(lists)

    fetch_start = time.perf_counter()
    futures = {thread_pool_global.submit(process_list, list_idx): list_idx 
                for list_idx in unique_lists}
    for future in as_completed(futures):
        mask, vectors = future.result()
        out[mask] = vectors
    print(f"Total fetch time: {time.perf_counter() - fetch_start:.4f} s")

    print(f"Total time: {time.perf_counter() - start_time:.4f} s")
    return out



# def get_vectors_server(ids: np.ndarray, emb_map: np.memmap, id_to_sorted_row_map: np.ndarray, current_layout_prefix: np.ndarray) -> np.ndarray:
#     """
#     Fetches full embedding vectors for given original vector IDs.
#     Assumes emb_map contains embeddings sorted list-wise.
#     Now uses global thread pool to parallelize list processing for better performance.
#     """
#     rows = id_to_sorted_row_map[ids]  # Convert original IDs to sorted row indices
#     # Determine which list each sorted row index belongs to
#     lists = np.searchsorted(current_layout_prefix, rows, 'right') - 1
#     out = np.empty((len(ids), VECTOR_DIM), dtype=np.float16)

#     def process_list(list_idx):
#         """Process a single list and return mask and vectors for that list."""
#         start_offset = current_layout_prefix[list_idx]
#         # Assumes current_layout_prefix has num_lists + 1 elements for the end boundary
#         end_offset = current_layout_prefix[list_idx + 1] 
#         mask = (lists == list_idx)
#         # Offsets within the specific list's block in emb_map
#         offsets_in_list_block = rows[mask] - start_offset
#         vectors = emb_map[start_offset:end_offset][offsets_in_list_block]
#         return mask, vectors

#     unique_lists = np.unique(lists)
    
#     # If only one list, no need for threading overhead
#     if len(unique_lists) == 1:
#         list_idx = unique_lists[0]
#         start_offset = current_layout_prefix[list_idx]
#         end_offset = current_layout_prefix[list_idx + 1] 
#         mask = (lists == list_idx)
#         offsets_in_list_block = rows[mask] - start_offset
#         out[mask] = emb_map[start_offset:end_offset][offsets_in_list_block]
#     else:
#         # Use global thread pool for multiple lists
#         futures = {thread_pool_global.submit(process_list, list_idx): list_idx 
#                   for list_idx in unique_lists}
        
#         # Collect results as they complete
#         for future in as_completed(futures):
#             mask, vectors = future.result()
#             out[mask] = vectors

#     return out # Returns float16 vectors

def get_metadata_lmdb_server(original_vector_ids: np.ndarray, txn: lmdb.Transaction, db: Any, id_to_sorted_row_map: np.ndarray):
    """
    Fetches metadata from LMDB for given original vector IDs.
    Optimized for potentially better sequential access by sorting keys.
    """
    items_to_fetch = []
    for i, vec_id in enumerate(original_vector_ids):
        lmdb_key = id_to_sorted_row_map[vec_id] 
        items_to_fetch.append((lmdb_key, i))

    items_to_fetch.sort(key=lambda x: x[0])

    results_in_original_order = [None] * len(original_vector_ids)
    total_get_time = 0.0
    total_decode_time = 0.0

    cur = txn.cursor(db=db)

    for lmdb_key_val, original_pos_idx in items_to_fetch:
        packed_lmdb_key = struct.pack("!Q", int(lmdb_key_val)) # Ensure Python int for struct.pack

        get_start_time = time.perf_counter()
        found = cur.set_key(packed_lmdb_key)
        binary_data_buffer = cur.value() if found else None
        get_end_time = time.perf_counter()
        total_get_time += (get_end_time - get_start_time)

        decode_start_time = time.perf_counter()
        if binary_data_buffer:
            try:
                # Assuming format: text_len (I), title_len (I), url_len (I), text (bytes), title (bytes), url (bytes)
                # We only need the text for now.
                text_len, _, _ = struct.unpack_from("!III", binary_data_buffer, 0) # title_len, url_len not used
                offset = 12 # Size of III (3*4 bytes)
                text_slice = binary_data_buffer[offset : offset + text_len]
                text = text_slice.tobytes().decode('utf-8', errors='replace')
                record = {'text': text}
            except Exception as e:
                logging.error(f"Error decoding LMDB data for key {lmdb_key_val}: {e}")
                record = {'text': f'Error decoding metadata for ID {lmdb_key_val}'}
        else:
            logging.warning(f"LMDB key {lmdb_key_val} (original vec_id {original_vector_ids[original_pos_idx]}) not found.")
            record = {'text': f'No metadata found for LMDB key {lmdb_key_val}'}
        decode_end_time = time.perf_counter()
        total_decode_time += (decode_end_time - decode_start_time)
        results_in_original_order[original_pos_idx] = record

    return results_in_original_order, total_get_time, total_decode_time

@app.on_event("startup")
async def startup_event():
    global index_global, cohere_client_global, embeddings_global, layout_prefix_global, \
           id_to_sorted_row_global, lmdb_env_global, lmdb_metadata_db_global, thread_pool_global

    logging.info("Server starting up...")
    startup_timer_start = time.perf_counter()

    if not COHERE_API_KEY:
        logging.error("COHERE_API_KEY not set. Please set the environment variable.")
        raise RuntimeError("COHERE_API_KEY not configured.")
    cohere_client_global = cohere.Client(COHERE_API_KEY)
    logging.info("Cohere client initialized.")

    # --- Load ID to sorted row map first (needed for total_embeddings and other loads) ---
    if not Path(ID_TO_SORTED_ROW_PATH).exists():
        logging.error(f"ID_TO_SORTED_ROW_PATH file not found: {ID_TO_SORTED_ROW_PATH}")
        raise RuntimeError(f"ID_TO_SORTED_ROW_PATH file not found: {ID_TO_SORTED_ROW_PATH}")
    id_to_sorted_row_global = np.load(ID_TO_SORTED_ROW_PATH, mmap_mode="r")
    total_embeddings = len(id_to_sorted_row_global)
    logging.info(f"ID to sorted row map loaded. Total items: {total_embeddings}")

    # --- Load IVF-PQ index ---
    if not INDEX_PATH.exists():
        logging.error(f"Index file not found: {INDEX_PATH}")
        raise RuntimeError(f"Index file not found: {INDEX_PATH}")
    index_global = ivf_pq.load(str(INDEX_PATH))
    logging.info("IVF-PQ index loaded.")

    # --- Load embeddings (RAM vs memory-mapped) ---
    if not Path(EMBEDDINGS_PATH).exists():
        logging.error(f"Embeddings file not found: {EMBEDDINGS_PATH}")
        raise RuntimeError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    
    # Calculate memory requirements
    embedding_size_gb = (total_embeddings * VECTOR_DIM * 2) / (1024**3)  # 2 bytes per float16
    
    if LOAD_EMBEDDINGS_TO_RAM:
        logging.info(f"Loading embeddings into RAM (requires ~{embedding_size_gb:.1f} GB)...")
        logging.warning(f"This will consume {embedding_size_gb:.1f} GB of system RAM!")
        
        # Load embeddings entirely into RAM for maximum performance
        embeddings_mmap = np.memmap(EMBEDDINGS_PATH, dtype=np.float16, mode="r", shape=(total_embeddings, VECTOR_DIM))
        embeddings_global = np.array(embeddings_mmap, dtype=np.float16)  # Copy to RAM
        del embeddings_mmap  # Free the memmap reference
        
        logging.info(f"Embeddings loaded into RAM ({embedding_size_gb:.1f} GB). Maximum performance mode enabled.")
    else:
        logging.info(f"Using memory-mapped embeddings (file size: ~{embedding_size_gb:.1f} GB)")
        logging.info("For better performance on systems with sufficient RAM, set LOAD_EMBEDDINGS_TO_RAM=true")
        
        # Use memory-mapped file (lazy loading, lower memory usage)
        embeddings_global = np.memmap(EMBEDDINGS_PATH, dtype=np.float16, mode="r", shape=(total_embeddings, VECTOR_DIM))
        
        logging.info("Embeddings memmap loaded (memory-efficient mode).")

    # --- Load IVF-PQ layout (prefix for get_vectors_server) ---
    if not Path(IVF_PQ_LAYOUT_PATH).exists():
        logging.error(f"IVF_PQ_LAYOUT_PATH file not found: {IVF_PQ_LAYOUT_PATH}")
        raise RuntimeError(f"IVF_PQ_LAYOUT_PATH file not found: {IVF_PQ_LAYOUT_PATH}")
    layout_data = np.load(IVF_PQ_LAYOUT_PATH)
    layout_prefix_global = layout_data["prefix"] # Contains cumulative sums of list sizes
    logging.info("IVF-PQ layout loaded.")

    # --- Open LMDB environment ---
    if not Path(LMDB_META_PATH).exists(): # LMDB_META_PATH is a directory
        logging.error(f"LMDB_META_PATH directory not found: {LMDB_META_PATH}")
        raise RuntimeError(f"LMDB_META_PATH directory not found {LMDB_META_PATH}")
    lmdb_env_global = lmdb.open(LMDB_META_PATH, readonly=True, lock=False, max_dbs=2, max_readers=200, readahead=True) # readahead=False for mmap
    lmdb_metadata_db_global = lmdb_env_global.open_db(b'metadata')
    logging.info("LMDB environment opened.")

    # --- Initialize thread pool for vector retrieval ---
    thread_pool_global = ThreadPoolExecutor(max_workers=8, thread_name_prefix="vector_fetch")
    logging.info("Thread pool initialized for vector retrieval.")

    # --- Warmup ---
    logging.info("Running warmup query...")
    try:
        warmup_query_text = "Initialize server components"
        # Ensure cohere_client_global is used, not co
        response = cohere_client_global.embed(texts=[warmup_query_text], model='multilingual-22-12', input_type='search_query')
        query_embedding_warmup = np.array(response.embeddings[0], dtype=np.float32)
        query_gpu_warmup = cp.asarray(query_embedding_warmup, dtype=cp.float32).reshape(1, VECTOR_DIM)
        
        search_params_warmup = ivf_pq.SearchParams(n_probes=N_PROBES_SEARCH)
        # Ensure index_global is used
        _, candidates_pq_warmup = ivf_pq.search(search_params_warmup, index_global, query_gpu_warmup, OVERSAMPLE_K)
        
        # Check if candidates_pq_warmup has valid dimensions before trying to access [0]
        if candidates_pq_warmup.shape[0] > 0 and candidates_pq_warmup.shape[1] > 0:
            candidates_cpu_warmup = cp.asnumpy(candidates_pq_warmup)[0]
            if candidates_cpu_warmup.size > 0:
                # Filter for valid candidates before further processing
                valid_mask = (candidates_cpu_warmup >= 0) & (candidates_cpu_warmup < len(id_to_sorted_row_global))
                valid_candidates_warmup = candidates_cpu_warmup[valid_mask]

                if valid_candidates_warmup.size > 0:
                    # Ensure all global variables are used for helper functions
                    _ = get_vectors_server(valid_candidates_warmup, embeddings_global, id_to_sorted_row_global, layout_prefix_global)
                    with lmdb_env_global.begin(db=lmdb_metadata_db_global, buffers=True) as txn_warmup:
                        _ = get_metadata_lmdb_server(valid_candidates_warmup[:FINAL_K], txn_warmup, lmdb_metadata_db_global, id_to_sorted_row_global)
                    logging.info("Warmup vector fetch and LMDB access successful.")
                else:
                    logging.warning("Warmup query: No valid candidates after filtering.")
            else:
                logging.warning("Warmup query: No candidates from IVF-PQ search (after asnumpy).")
        else:
            logging.warning("Warmup query: IVF-PQ search returned empty or unexpectedly shaped tensor.")

        # Touch LMDB pages
        logging.info("[WARMUP_LMD_TOUCH] Attempting to touch LMDB pages...")
        try:
            with lmdb_env_global.begin(db=lmdb_metadata_db_global, buffers=True) as txn_touch:
                logging.info("[WARMUP_LMD_TOUCH] Transaction for page touching started.")
                cur_touch = txn_touch.cursor()
                logging.info("[WARMUP_LMD_TOUCH] Cursor created. Starting iteration...")
                count = 0
                iter_start_time = time.perf_counter()
                for _ in cur_touch: 
                    count += 1
                    if count % 5_000_000 == 0: # Log progress every 5 million records
                        logging.info(f"[WARMUP_LMD_TOUCH] Iterated over {count} records...")
                iter_end_time = time.perf_counter()
                logging.info(f"[WARMUP_LMD_TOUCH] Finished iterating over {count} records. Time taken: {iter_end_time - iter_start_time:.2f}s")
            logging.info("LMDB pages touched (warmup).") # This is the original success log for this step
        except Exception as e_touch:
            logging.error(f"[WARMUP_LMD_TOUCH] Error during LMDB page touching: {e_touch}", exc_info=True)
            # Re-raise or handle as needed; for now, just logging it within the warmup try-block context

    except Exception as e:
        logging.error(f"Error during warmup: {e}", exc_info=True)
        # Don't raise here, allow server to start if possible, but log failure.

    # Touch embeddings pages to pre-load into memory (only for memory-mapped files)
    if not LOAD_EMBEDDINGS_TO_RAM:
        logging.info("Pre-loading embeddings memmap into memory...")
        touch_start_time = time.perf_counter()

        for i in range(0, total_embeddings, 1000000):
            start_idx = i
            end_idx = min(start_idx + 1000000, total_embeddings)
            _ = embeddings_global[start_idx:end_idx]
            if i % 1000000 == 0:  # Log progress
                logging.info(f"Touched {i:,} vectors...")

        touch_end_time = time.perf_counter()
        logging.info(f"Embeddings memmap pre-loading completed in {touch_end_time - touch_start_time:.2f}s")
    else:
        logging.info("Skipping embeddings pre-loading (already in RAM)")

    logging.info(f"Server startup is completed. Total time: {time.perf_counter() - startup_timer_start:.4f} seconds")


@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Server shutting down...")
    if thread_pool_global:
        thread_pool_global.shutdown(wait=True)
        logging.info("Thread pool shut down.")
    if lmdb_env_global:
        lmdb_env_global.close()
        logging.info("LMDB environment closed.")
    # CuPy context cleanup is usually automatic upon process exit.
    # Explicit cleanup can be added if specific issues arise, e.g.:
    # cp.get_default_memory_pool().free_all_blocks()
    # cp.get_default_pinned_memory_pool().free_all_blocks()
    logging.info("Server shutdown complete.")


@app.get("/health")
async def health_check():
    """Health check endpoint for orchestrator"""
    # Check if all global resources are initialized
    expected_globals = [
        index_global, cohere_client_global, embeddings_global, 
        id_to_sorted_row_global, layout_prefix_global, 
        lmdb_env_global, lmdb_metadata_db_global
    ]
    
    if any(item is None for item in expected_globals):
        raise HTTPException(status_code=503, detail="Server not fully initialized")
    
    return {"status": "healthy", "message": "Semantic search server is ready"}

@app.get("/status")
async def status_check():
    """Detailed status endpoint"""
    # Check if all global resources are initialized
    expected_globals = [
        index_global, cohere_client_global, embeddings_global, 
        id_to_sorted_row_global, layout_prefix_global, 
        lmdb_env_global, lmdb_metadata_db_global
    ]
    
    global_names = [
        "index_global", "cohere_client_global", "embeddings_global",
        "id_to_sorted_row_global", "layout_prefix_global",
        "lmdb_env_global", "lmdb_metadata_db_global"
    ]
    
    status = {}
    for i, item in enumerate(expected_globals):
        status[global_names[i]] = "initialized" if item is not None else "not_initialized"
    
    all_ready = all(item is not None for item in expected_globals)
    
    return {
        "status": "ready" if all_ready else "initializing",
        "components": status,
        "ready": all_ready
    }

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query_processing_start_time = time.perf_counter()
    timings = {}
    user_query = request.query
    k_final = request.top_k
    logging.info(f"--- Received query: '{user_query}', top_k: {k_final} ---")

    # Check if all global resources are initialized
    expected_globals = [
        index_global, cohere_client_global, embeddings_global, 
        id_to_sorted_row_global, layout_prefix_global, 
        lmdb_env_global, lmdb_metadata_db_global
    ]
    if any(item is None for item in expected_globals):
        # For more detailed logging, find out which one is None
        uninitialized_globals = []
        global_names = [
            "index_global", "cohere_client_global", "embeddings_global",
            "id_to_sorted_row_global", "layout_prefix_global",
            "lmdb_env_global", "lmdb_metadata_db_global"
        ]
        for i, item in enumerate(expected_globals):
            if item is None:
                uninitialized_globals.append(global_names[i])
        logging.error(f"Server not fully initialized. The following global resources are None: {', '.join(uninitialized_globals)}")
        raise HTTPException(status_code=503, detail="Server is not fully initialized. Please try again later.")

    try:
        # 1. Get embedding from Cohere
        step_start = time.perf_counter()
        # Specify input_type for Cohere v3+ embed API
        response = cohere_client_global.embed(texts=[user_query], model='multilingual-22-12', input_type='search_query')
        query_embedding = np.array(response.embeddings[0], dtype=np.float32)
        timings['cohere_embedding_time'] = time.perf_counter() - step_start

        # 2. Move query embedding to GPU
        step_start = time.perf_counter()
        query_gpu = cp.asarray(query_embedding, dtype=cp.float32).reshape(1, VECTOR_DIM)
        timings['gpu_transfer_time'] = time.perf_counter() - step_start

        # 3. IVF-PQ search
        step_start = time.perf_counter()
        search_params = ivf_pq.SearchParams(n_probes=N_PROBES_SEARCH)
        # Distances are not used here, only candidate IDs
        _, candidates_pq = ivf_pq.search(search_params, index_global, query_gpu, OVERSAMPLE_K)
        
        # Check if candidates_pq has valid dimensions before trying to access [0]
        if not (candidates_pq.shape[0] > 0 and candidates_pq.shape[1] > 0):
            logging.warning(f"IVF-PQ search returned no candidates or unexpectedly shaped tensor for query: {user_query}")
            return QueryResponse(results=[], timings=timings)
        
        candidates_cpu_original_ids = cp.asnumpy(candidates_pq)[0] # These are original vector IDs
        timings['ivfpq_search_time'] = time.perf_counter() - step_start

        if candidates_cpu_original_ids.size == 0:
            logging.warning(f"No candidates after IVF-PQ search (asnumpy) for query: {user_query}")
            return QueryResponse(results=[], timings=timings)

        # Filter for valid candidate IDs (non-negative and within bounds of id_to_sorted_row_global)
        valid_candidates_mask = (candidates_cpu_original_ids >= 0) & (candidates_cpu_original_ids < len(id_to_sorted_row_global))
        valid_candidates_original_ids = candidates_cpu_original_ids[valid_candidates_mask]

        if valid_candidates_original_ids.size == 0:
            logging.warning(f"No valid candidates after filtering for query: {user_query}")
            return QueryResponse(results=[], timings=timings)
            
        # 4. Fetch candidate vectors for reranking
        step_start = time.perf_counter()
        candidate_vecs_fp16 = get_vectors_server(valid_candidates_original_ids, embeddings_global, id_to_sorted_row_global, layout_prefix_global)
        timings['vector_fetch_time'] = time.perf_counter() - step_start

        # 5. Rerank with exact dot product
        step_start = time.perf_counter()
        # query_embedding is float32, candidate_vecs_fp16 is float16. np.dot handles mixed precision.
        scores = np.dot(candidate_vecs_fp16, query_embedding)
        
        # Determine how many results to actually return (might be less than k_final if fewer candidates)
        actual_k_to_return = min(k_final, len(scores))
        if actual_k_to_return == 0: # Should not happen if valid_candidates_original_ids was not empty
             logging.warning(f"No scores to rank for query (should be impossible if candidates existed): {user_query}")
             return QueryResponse(results=[], timings=timings)

        # Get indices of top-k scores from the 'scores' array (which corresponds to 'valid_candidates_original_ids')
        topk_local_indices = np.argsort(scores)[-actual_k_to_return:][::-1]
        
        # Use these local indices to get the final original IDs and their scores
        final_top_original_ids = valid_candidates_original_ids[topk_local_indices]
        final_top_scores = scores[topk_local_indices]
        timings['reranking_time'] = time.perf_counter() - step_start

        # 6. Fetch metadata from LMDB
        step_start = time.perf_counter()
        # A new transaction should be started for each query for thread-safety if using multi-threading for web server.
        # For FastAPI async, this is fine as LMDB is opened read-only.
        with lmdb_env_global.begin(db=lmdb_metadata_db_global, buffers=True) as txn_query:
            metadata_batch, batch_get_time, batch_decode_time = get_metadata_lmdb_server(
                final_top_original_ids, txn_query, lmdb_metadata_db_global, id_to_sorted_row_global
            )
        timings['lmdb_metadata_fetch_total_time'] = time.perf_counter() - step_start
        timings['lmdb_raw_get_time'] = batch_get_time
        timings['lmdb_decode_time'] = batch_decode_time
        
        # 7. Format results
        output_results = []
        for rank, (item, score) in enumerate(zip(metadata_batch, final_top_scores), 1):
            output_results.append(ResultItem(rank=rank, score=float(score), text=item['text']))

        total_query_processing_time = time.perf_counter() - query_processing_start_time
        timings['total_query_processing_time'] = total_query_processing_time
        
        # Create simplified timing response (convert to milliseconds and clean names)
        simple_timings = {
            "embedding_ms": round(timings['cohere_embedding_time'] * 1000, 1),
            "search_ms": round(timings['ivfpq_search_time'] * 1000, 1),
            "retrieval_ms": round(timings['vector_fetch_time'] * 1000, 1),
            "reranking_ms": round(timings['reranking_time'] * 1000, 1),
            "metadata_ms": round(timings['lmdb_metadata_fetch_total_time'] * 1000, 1),
            "total_ms": round(total_query_processing_time * 1000, 1)
        }
        
        # Detailed logging (keep original format for logs)
        logging.info(f"--- Query Timings for: '{user_query}' ---")
        logging.info(f"  Embedding: {simple_timings['embedding_ms']}ms")
        logging.info(f"  Search: {simple_timings['search_ms']}ms")
        logging.info(f"  Retrieval: {simple_timings['retrieval_ms']}ms")
        logging.info(f"  Reranking: {simple_timings['reranking_ms']}ms")
        logging.info(f"  Metadata: {simple_timings['metadata_ms']}ms")
        logging.info(f"  Total: {simple_timings['total_ms']}ms")
        
        return QueryResponse(results=output_results, timings=simple_timings)

    except cohere.CohereAPIError as e:
        logging.error(f"Cohere API error for query '{user_query}': {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error communicating with embedding service: {str(e)}")
    except lmdb.Error as e: # More specific LMDB errors can be caught if needed e.g. lmdb.KeyExistsError etc.
        logging.error(f"LMDB error for query '{user_query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error accessing metadata database: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error processing query '{user_query}': {e}", exc_info=True)
        # It's good practice to not expose raw internal error messages to the client.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Main block to run the server (for direct execution) ---
if __name__ == "__main__":
    if not COHERE_API_KEY:
        print("ERROR: COHERE_API_KEY environment variable not set.")
        print("Please ensure the config.env file exists with your Cohere API key.")
        print("Example config.env content:")
        print("COHERE_API_KEY=your_cohere_api_key_here")
        exit(1)
    
    # Default host="127.0.0.1" makes it only accessible locally.
    # Use host="0.0.0.0" to make it accessible from the network (e.g., within Docker or LAN).
    uvicorn.run(app, host="0.0.0.0", port=8000) 