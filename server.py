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
import asyncio
import threading

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

# Global variables for components
index_global = None
cohere_client_global = None
embeddings_global = None
layout_prefix_global = None
id_to_sorted_row_global = None
lmdb_env_global = None
lmdb_metadata_db_global = None
thread_pool_global = None

# Global variables for initialization tracking
initialization_status = {
    "stage": "starting",
    "progress": 0,
    "message": "Server starting up...",
    "error": None,
    "ready": False,
    "start_time": None,
    "stages": {
        "cohere_client": {"status": "pending", "message": "Initializing Cohere client"},
        "id_mapping": {"status": "pending", "message": "Loading ID to sorted row mapping"},
        "index": {"status": "pending", "message": "Loading IVF-PQ index"},
        "embeddings": {"status": "pending", "message": "Loading embeddings"},
        "layout": {"status": "pending", "message": "Loading IVF-PQ layout"},
        "lmdb": {"status": "pending", "message": "Opening LMDB environment"},
        "lmdb_page_warmup": {"status": "pending", "message": "Warming up LMDB pages"},
        "thread_pool": {"status": "pending", "message": "Initializing thread pool"},
        "app_warmup": {"status": "pending", "message": "Running application warmup queries"}
    }
}
initialization_lock = threading.Lock()

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
    """
    Fetches full embedding vectors for given original vector IDs.
    Uses simple numpy operations for better performance with typical small numbers of unique lists.
    """
    rows = id_to_sorted_row_map[ids]
    lists = np.searchsorted(current_layout_prefix, rows, 'right') - 1
    out = np.empty((len(ids), VECTOR_DIM), dtype=np.float16)
    
    unique_lists = np.unique(lists)
    
    for list_idx in unique_lists:
        mask = (lists == list_idx)
        # Direct fancy indexing - no intermediate slice copy
        out[mask] = emb_map[rows[mask]]
    
    return out

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

def update_initialization_status(stage: str = None, progress: int = None, message: str = None, error: str = None, ready: bool = None):
    """Update the global initialization status"""
    with initialization_lock:
        if stage:
            initialization_status["stage"] = stage
        if progress is not None:
            initialization_status["progress"] = progress
        if message:
            initialization_status["message"] = message
        if error:
            initialization_status["error"] = error
        if ready is not None:
            initialization_status["ready"] = ready

def update_stage_status(stage_name: str, status: str, message: str = None):
    """Update the status of a specific initialization stage"""
    with initialization_lock:
        if stage_name in initialization_status["stages"]:
            initialization_status["stages"][stage_name]["status"] = status
            if message:
                initialization_status["stages"][stage_name]["message"] = message

async def initialize_server_async():
    """Asynchronous server initialization that runs in background"""
    global index_global, cohere_client_global, embeddings_global, layout_prefix_global, \
           id_to_sorted_row_global, lmdb_env_global, lmdb_metadata_db_global, thread_pool_global

    try:
        initialization_status["start_time"] = time.time()
        logging.info("Starting asynchronous server initialization...")
        startup_timer_start = time.perf_counter()

        # --- Sequential Part 1 ---
        # Stage 1: Cohere Client
        update_initialization_status(stage="cohere_client", progress=5, message="Initializing Cohere client...")
        update_stage_status("cohere_client", "in_progress")
        if not COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY not configured.")
        cohere_client_global = cohere.Client(COHERE_API_KEY)
        update_stage_status("cohere_client", "completed", "Cohere client initialized")
        logging.info("Cohere client initialized.")

        # Stage 2: ID Mapping
        update_initialization_status(stage="id_mapping", progress=10, message="Loading ID to sorted row mapping...")
        update_stage_status("id_mapping", "in_progress")
        if not Path(ID_TO_SORTED_ROW_PATH).exists(): raise RuntimeError(f"ID_TO_SORTED_ROW_PATH file not found: {ID_TO_SORTED_ROW_PATH}")
        def _load_id_map_sync():
            logging.info(f"Worker thread: Loading ID to sorted row map from {ID_TO_SORTED_ROW_PATH}...")
            loaded_map = np.load(ID_TO_SORTED_ROW_PATH)
            map_size_mb = loaded_map.nbytes / (1024**2)
            logging.info(f"Worker thread: ID to sorted row map loaded into RAM ({map_size_mb:.1f} MB).")
            return loaded_map
        id_to_sorted_row_global = await asyncio.to_thread(_load_id_map_sync)
        total_embeddings = len(id_to_sorted_row_global)
        update_stage_status("id_mapping", "completed", f"ID mapping loaded. Total items: {total_embeddings:,}")
        logging.info(f"ID to sorted row map loaded. Total items: {total_embeddings}")

        # Stage 3: Index
        update_initialization_status(stage="index", progress=15, message="Loading IVF-PQ index...")
        update_stage_status("index", "in_progress")
        if not INDEX_PATH.exists(): raise RuntimeError(f"Index file not found: {INDEX_PATH}")
        def _load_index_sync():
            logging.info("Worker thread: Loading IVF-PQ index...")
            idx = ivf_pq.load(str(INDEX_PATH))
            logging.info("Worker thread: IVF-PQ index loaded.")
            return idx
        index_global = await asyncio.to_thread(_load_index_sync)
        update_stage_status("index", "completed", "IVF-PQ index loaded")
        logging.info("IVF-PQ index loaded.")

        # Stage 5: Layout (Original stage number, keeping sequence)
        update_initialization_status(stage="layout", progress=20, message="Loading IVF-PQ layout...")
        update_stage_status("layout", "in_progress")
        if not Path(IVF_PQ_LAYOUT_PATH).exists(): raise RuntimeError(f"IVF_PQ_LAYOUT_PATH file not found: {IVF_PQ_LAYOUT_PATH}")
        def _load_layout_sync():
            logging.info("Worker thread: Loading IVF-PQ layout...")
            layout_data = np.load(IVF_PQ_LAYOUT_PATH)
            logging.info("Worker thread: IVF-PQ layout loaded.")
            return layout_data["prefix"]
        layout_prefix_global = await asyncio.to_thread(_load_layout_sync)
        update_stage_status("layout", "completed", "IVF-PQ layout loaded")
        logging.info("IVF-PQ layout loaded.")

        # Stage 6: LMDB Open
        update_initialization_status(stage="lmdb", progress=25, message="Opening LMDB environment...")
        update_stage_status("lmdb", "in_progress")
        if not Path(LMDB_META_PATH).exists(): raise RuntimeError(f"LMDB_META_PATH directory not found {LMDB_META_PATH}")
        def _open_lmdb_sync():
            logging.info("Worker thread: Opening LMDB environment...")
            env = lmdb.open(LMDB_META_PATH, readonly=True, lock=False, max_dbs=2, max_readers=200, readahead=True)
            db = env.open_db(b'metadata')
            logging.info("Worker thread: LMDB environment opened.")
            return env, db
        lmdb_env_global, lmdb_metadata_db_global = await asyncio.to_thread(_open_lmdb_sync)
        update_stage_status("lmdb", "completed", "LMDB environment opened")
        logging.info("LMDB environment opened.")

        # --- Parallel Part: Embeddings (Stage 4) and LMDB Page Warmup ---
        update_initialization_status(stage="parallel_heavy_loading", progress=30, 
                                     message="Concurrently: Loading embeddings & Warming LMDB pages...")

        # Embedding loading helpers (defined here to capture total_embeddings)
        embedding_size_gb = (total_embeddings * VECTOR_DIM * 2) / (1024**3)
        def _load_embeddings_to_ram_sync():
            logging.info(f"Worker thread (Embeddings): Loading into RAM (~{embedding_size_gb:.1f} GB)...")
            mmap_view = np.memmap(EMBEDDINGS_PATH, dtype=np.float16, mode="r", shape=(total_embeddings, VECTOR_DIM))
            
            copy_start_time = time.perf_counter()
            embeddings_ram = mmap_view.copy(order="C")
            copy_duration = time.perf_counter() - copy_start_time
            logging.info(f"WORKER THREAD (EmbeddingsLoadToRAM): mmap_view.copy() took {copy_duration:.2f}s.")
            
            mmap_view._mmap.close()
            del mmap_view
            logging.info(f"Worker thread (Embeddings): Loaded into RAM.")
            return embeddings_ram

        def _memmap_embeddings_sync():
            logging.info(f"Worker thread (Embeddings): Memory-mapping (~{embedding_size_gb:.1f} GB)...")
            emb = np.memmap(EMBEDDINGS_PATH, dtype=np.float16, mode="r", shape=(total_embeddings, VECTOR_DIM))
            logging.info(f"Worker thread (Embeddings): Memory-mapped.")
            return emb

        # LMDB page warmup helper (already defined, uses lmdb_env_global from above)
        def _initialize_lmdb_page_warmup_sync(): # Already defined from previous step
            task_start_time = time.perf_counter()
            logging.info("Initialization (thread LMDB Warmup): Touching LMDB pages...")
            count = 0
            loop_duration = 0.0
            try:
                with lmdb_env_global.begin(db=lmdb_metadata_db_global, buffers=True) as txn_touch:
                    cur_touch = txn_touch.cursor()
                    
                    loop_start_time = time.perf_counter()
                    for _ in cur_touch:
                        count += 1
                        if count % 5_000_000 == 0:
                            logging.info(f"Initialization (thread LMDB Warmup): Touched {count:,} LMDB records...")
                    loop_duration = time.perf_counter() - loop_start_time
                    logging.info(f"WORKER THREAD (LMDBWarmup): Iteration loop took {loop_duration:.2f}s.")

            except Exception as e_lmdb_touch:
                logging.error(f"Initialization (thread LMDB Warmup): Error during page touching: {e_lmdb_touch}", exc_info=True)
            
            total_function_duration = time.perf_counter() - task_start_time
            logging.info(f"WORKER THREAD (LMDBWarmup): Finished page touching. Touched {count:,} records. Loop: {loop_duration:.2f}s. Total function time: {total_function_duration:.2f}s.")
            return count

        # Create tasks for parallel execution
        tasks_to_run_in_parallel = []

        update_stage_status("embeddings", "in_progress", f"Loading embeddings ({embedding_size_gb:.1f} GB)... ")
        if not Path(EMBEDDINGS_PATH).exists(): raise RuntimeError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
        if LOAD_EMBEDDINGS_TO_RAM:
            task_embeddings = asyncio.create_task(asyncio.to_thread(_load_embeddings_to_ram_sync), name="load_embeddings_ram")
        else:
            task_embeddings = asyncio.create_task(asyncio.to_thread(_memmap_embeddings_sync), name="memmap_embeddings")
        tasks_to_run_in_parallel.append(task_embeddings)

        update_stage_status("lmdb_page_warmup", "in_progress", "Warming up LMDB pages...")
        task_lmdb_warmup = asyncio.create_task(asyncio.to_thread(_initialize_lmdb_page_warmup_sync), name="lmdb_page_warmup")
        tasks_to_run_in_parallel.append(task_lmdb_warmup)
        
        logging.info(f"Starting parallel execution of: {', '.join(t.get_name() for t in tasks_to_run_in_parallel)}")
        results = await asyncio.gather(*tasks_to_run_in_parallel, return_exceptions=True)
        logging.info("Parallel loading/warming tasks completed.")

        # Process results (Embeddings is first in tasks_to_run_in_parallel)
        embeddings_result = results[0]
        if isinstance(embeddings_result, Exception):
            logging.error(f"Failed to load/memmap embeddings: {embeddings_result}", exc_info=embeddings_result)
            raise RuntimeError(f"Failed to load/memmap embeddings: {embeddings_result}")
        embeddings_global = embeddings_result
        if LOAD_EMBEDDINGS_TO_RAM:
            update_stage_status("embeddings", "completed", f"Embeddings loaded into RAM ({embedding_size_gb:.1f} GB)")
        else:
            update_stage_status("embeddings", "completed", f"Embeddings memmap loaded ({embedding_size_gb:.1f} GB)")
        
        # Process LMDB warmup result (second in tasks_to_run_in_parallel)
        lmdb_warmup_result = results[1]
        if isinstance(lmdb_warmup_result, Exception):
            # This might not be fatal for server startup, log error and continue, or raise if critical
            logging.error(f"LMDB page warmup failed or encountered an error: {lmdb_warmup_result}", exc_info=lmdb_warmup_result)
            update_stage_status("lmdb_page_warmup", "error", f"LMDB page warmup error: {lmdb_warmup_result}")
            # Decide if this is a fatal error. For now, we'll allow server to continue.
        else:
            update_stage_status("lmdb_page_warmup", "completed", "LMDB page warmup finished")
        logging.info("LMDB page warmup processing finished.") # General log regardless of outcome
        
        # --- Sequential Part 2 ---
        update_initialization_status(stage="thread_pool", progress=85, message="Initializing thread pool...") # Adjusted progress
        update_stage_status("thread_pool", "in_progress")
        thread_pool_global = ThreadPoolExecutor(max_workers=8, thread_name_prefix="vector_fetch")
        update_stage_status("thread_pool", "completed", "Thread pool initialized")
        logging.info("Thread pool initialized for vector retrieval.")

        update_initialization_status(stage="app_warmup", progress=90, message="Running application warmup queries...")
        update_stage_status("app_warmup", "in_progress")
        await run_warmup() # run_warmup (app_warmup) remains unchanged
        update_stage_status("app_warmup", "completed", "Warmup queries completed")

        total_time = time.perf_counter() - startup_timer_start
        update_initialization_status(stage="ready", progress=100, 
                                    message=f"Server ready! Initialization completed in {total_time:.1f}s", 
                                    ready=True)
        logging.info(f"Server startup completed. Total time: {total_time:.4f} seconds")

    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        update_initialization_status(error=error_msg, stage="error", progress=initialization_status.get("progress", 0))
        logging.error(f"Server initialization failed: {e}", exc_info=True)

async def run_warmup():
    """Run warmup queries"""
    try:
        warmup_query_text = "Initialize server components"

        # Cohere embed
        def _warmup_cohere_embed_sync():
            logging.info("AppWarmup (thread): Running Cohere embed...") # Changed log prefix
            response = cohere_client_global.embed(texts=[warmup_query_text], model='multilingual-22-12', input_type='search_query')
            logging.info("AppWarmup (thread): Cohere embed complete.")
            return np.array(response.embeddings[0], dtype=np.float32)
        query_embedding_warmup = await asyncio.to_thread(_warmup_cohere_embed_sync)
        
        query_gpu_warmup = cp.asarray(query_embedding_warmup, dtype=cp.float32).reshape(1, VECTOR_DIM)
        
        search_params_warmup = ivf_pq.SearchParams(n_probes=N_PROBES_SEARCH)
        def _warmup_ivf_pq_search_sync():
            logging.info("AppWarmup (thread): Running IVF-PQ search...") # Changed log prefix
            distances, candidates = ivf_pq.search(search_params_warmup, index_global, query_gpu_warmup, OVERSAMPLE_K)
            logging.info("AppWarmup (thread): IVF-PQ search complete.")
            return distances, candidates
        _, candidates_pq_warmup = await asyncio.to_thread(_warmup_ivf_pq_search_sync)
        
        if candidates_pq_warmup.shape[0] > 0 and candidates_pq_warmup.shape[1] > 0:
            def _warmup_asnumpy_sync():
                logging.info("AppWarmup (thread): Converting candidates to numpy...") # Changed log prefix
                res = cp.asnumpy(candidates_pq_warmup)[0]
                logging.info("AppWarmup (thread): Conversion to numpy complete.")
                return res
            candidates_cpu_warmup = await asyncio.to_thread(_warmup_asnumpy_sync)

            if candidates_cpu_warmup.size > 0:
                valid_mask = (candidates_cpu_warmup >= 0) & (candidates_cpu_warmup < len(id_to_sorted_row_global))
                valid_candidates_warmup = candidates_cpu_warmup[valid_mask]

                if valid_candidates_warmup.size > 0:
                    def _warmup_get_vectors_sync():
                        logging.info("AppWarmup (thread): Getting vectors...") # Changed log prefix
                        vecs = get_vectors_server(valid_candidates_warmup, embeddings_global, id_to_sorted_row_global, layout_prefix_global)
                        logging.info("AppWarmup (thread): Got vectors.")
                        return vecs
                    _ = await asyncio.to_thread(_warmup_get_vectors_sync)

                    def _warmup_get_metadata_sync():
                        logging.info("AppWarmup (thread): Getting metadata from LMDB...") # Changed log prefix
                        with lmdb_env_global.begin(db=lmdb_metadata_db_global, buffers=True) as txn_warmup:
                            meta = get_metadata_lmdb_server(valid_candidates_warmup[:FINAL_K], txn_warmup, lmdb_metadata_db_global, id_to_sorted_row_global)
                        logging.info("AppWarmup (thread): Got metadata from LMDB.")
                        return meta
                    _ = await asyncio.to_thread(_warmup_get_metadata_sync)

        # LMDB page touching logic has been removed from here

    except Exception as e:
        # Renamed stage to app_warmup
        logging.error(f"Error during application warmup (app_warmup): {e}", exc_info=True)
        # Optionally re-raise or update status if app_warmup failure is critical
        # For now, just log, as server might still be usable without full app warmup.

@app.on_event("startup")
async def startup_event():
    """Start the server and begin initialization in background"""
    # Start initialization in background
    asyncio.create_task(initialize_server_async())


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
    """Enhanced health check endpoint that reports initialization progress"""
    with initialization_lock:
        status_copy = initialization_status.copy()
        stages_copy = {k: v.copy() for k, v in initialization_status["stages"].items()}
        status_copy["stages"] = stages_copy
    
    # If there's an error, return 503
    if status_copy.get("error"):
        raise HTTPException(
            status_code=503, 
            detail={
                "status": "error",
                "error": status_copy["error"],
                "progress": status_copy["progress"],
                "stage": status_copy["stage"]
            }
        )
    
    # If not ready yet, return 503 with progress info
    if not status_copy.get("ready", False):
        return {
            "status": "initializing",
            "progress": status_copy["progress"],
            "stage": status_copy["stage"],
            "message": status_copy["message"],
            "stages": status_copy["stages"],
            "elapsed_time": time.time() - status_copy["start_time"] if status_copy["start_time"] else 0
        }
    
    # Server is ready
    return {
        "status": "ready",
        "progress": 100,
        "message": "Semantic search server is ready",
        "ready": True,
        "total_initialization_time": time.time() - status_copy["start_time"] if status_copy["start_time"] else 0
    }

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
        # Server is not fully initialized - provide detailed status like health check
        with initialization_lock:
            status_copy = initialization_status.copy()
        
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
        logging.error(f"Query attempted while server not fully initialized. Missing: {', '.join(uninitialized_globals)}")
        
        # Return detailed initialization status
        elapsed_time = time.time() - status_copy["start_time"] if status_copy.get("start_time") else 0
        
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Server is still initializing",
                "status": "initializing",
                "progress": status_copy.get("progress", 0),
                "stage": status_copy.get("stage", "unknown"),
                "message": status_copy.get("message", "Server is starting up..."),
                "elapsed_time": elapsed_time,
                "estimated_remaining": "Please check /health endpoint for real-time progress",
                "missing_components": uninitialized_globals
            }
        )

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