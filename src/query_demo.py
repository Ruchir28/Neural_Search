import cupy as cp
import numpy as np
from pathlib import Path
from cuvs.neighbors import ivf_pq
# from datasets import load_dataset, load_from_disk # Keep for reference if needed
import cohere
from train_index import VECTOR_DIM, DATASET, CACHE_DIR # DATASET and CACHE_DIR might be unused if dataset loading is removed
import time
import logging
import lmdb
import struct
import os
from dotenv import load_dotenv

# Load environment variables from config.env file
load_dotenv('../config.env')

# --- Configurable parameters ---
INDEX_PATH = Path("../trained_indices/trained_ivfpq_index_full.bin")
BATCH_SIZE = 100_000  # For batch loading if needed
OVERSAMPLE_K = 1000
FINAL_K = 15
N_PROBES_SEARCH = 40
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMBEDDINGS_PATH = "lists_info/embeddings_listwise.memmap"
IVF_PQ_LAYOUT_PATH = "lists_info/layout_sorted.npz"
ID_TO_SORTED_ROW_PATH = "lists_info/vec_id_to_sorted_row.bin.npy"
LMDB_META_PATH = "lists_info/metadata.lmdb"  # Path to the LMDB metadata

id_to_sorted_row = np.load(ID_TO_SORTED_ROW_PATH,mmap_mode="r")

# --- Setup logging ---
logging.basicConfig(filename='query_timing.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# --- Start overall timer ---
overall_start = time.perf_counter()

# --- Load the trained index ---
step_start = time.perf_counter()
print(f"Loading trained index from {INDEX_PATH}...")
index = ivf_pq.load(str(INDEX_PATH))

print("Index loaded.")
step_end = time.perf_counter()
logging.info(f"Index load time: {step_end - step_start:.4f} seconds")

# --- Open LMDB environment for metadata ---
step_start = time.perf_counter()
print(f"Opening LMDB environment at {LMDB_META_PATH}...")
# Ensure max_readers is set as per user's modification
env = lmdb.open(LMDB_META_PATH, readonly=True, lock=False, max_dbs=2, max_readers=100)
metadata_db = env.open_db(b'metadata')
txn = env.begin(buffers=True)
print("LMDB environment opened.")
step_end = time.perf_counter()
logging.info(f"LMDB open time: {step_end - step_start:.4f} seconds")

# --- Dataset loading commented out ---
# step_start = time.perf_counter()
# print(f"Loading sorted metadata from Parquet file...")
# dataset = load_dataset("parquet", data_files="lists_info/docs_sorted.parquet", split="train")
# dataset = load_from_disk("lists_info/meta_sorted")
# dataset = load_dataset(DATASET, split="train", cache_dir=CACHE_DIR) # Original dataset load
# step_end = time.perf_counter()
# logging.info(f"Dataset load time: {step_end - step_start:.4f} seconds")


total_embeddings = len(id_to_sorted_row)

emb = np.memmap(EMBEDDINGS_PATH, dtype=np.float16, mode="r", shape=(total_embeddings, 768))
layout = np.load(IVF_PQ_LAYOUT_PATH)
sizes: np.ndarray = layout["sizes"]
prefix: np.ndarray = layout["prefix"]

print(f"Total embeddings in dataset: {total_embeddings}")

cur = txn.cursor(db=metadata_db)
for _ in cur: pass  # walks every key, touching every page



# --- Setup Cohere client ---
co = cohere.Client(COHERE_API_KEY)

# --- Fetch candidate vectors for reranking ---
def get_vectors(ids,emb):
    rows  = id_to_sorted_row[ids]                       # (k,)
    lists = np.searchsorted(prefix, rows, 'right') - 1  # list ID per row
    out   = np.empty((len(ids), 768), np.float16)

    for L in np.unique(lists):
        s, e   = prefix[L], prefix[L + 1]       # contiguous block in emb
        mask   = (lists == L)
        offsets = rows[mask] - s                # row-offset *inside* that block
        out[mask] = emb[s:e][offsets]           # one slice, then fancy-index

    return out

# Removed get_metadata_from_dataset as dataset loading is commented out

def get_metadata_lmdb(original_vector_ids):
    """
    Get metadata using LMDB, optimized for potentially better sequential access
    and using buffers=True for potentially reduced data copying.
    Returns results and detailed timings for get and decode operations.
    """
    items_to_fetch = []
    for i, vec_id in enumerate(original_vector_ids):
        lmdb_key = id_to_sorted_row[vec_id]
        items_to_fetch.append((lmdb_key, i))

    items_to_fetch.sort(key=lambda x: x[0])

    results_in_original_order = [None] * len(original_vector_ids)
    total_get_time_batch = 0.0
    total_decode_time_batch = 0.0

    cur = txn.cursor(db=metadata_db)

    for lmdb_key_val, original_pos_idx in items_to_fetch:
        packed_lmdb_key = struct.pack("!Q", lmdb_key_val)

        
        get_start_time = time.perf_counter()
        cur.set_key(packed_lmdb_key)
        binary_data_buffer = cur.value()
        get_end_time = time.perf_counter()
        total_get_time_batch += (get_end_time - get_start_time)
        
        record = None
        decode_start_time = time.perf_counter()
        if binary_data_buffer:
            try:
                text_len, title_len, url_len = struct.unpack_from("!III", binary_data_buffer, 0)
                offset = 12 
                text_slice = binary_data_buffer[offset : offset + text_len]
                text = text_slice.tobytes().decode('utf-8', errors='replace')
                record = {'text': text, 'title': '', 'url': ''}
            except Exception as e:
                logging.error(f"Error decoding LMDB data for key {lmdb_key_val} (buffer mode): {e}")
                record = {'text': f'Error decoding metadata for ID {lmdb_key_val}', 'title': '', 'url': ''}
        else:
            logging.warning(f"LMDB key {lmdb_key_val} (original vec_id {original_vector_ids[original_pos_idx]}) not found.")
            record = {'text': f'No metadata found for LMDB key {lmdb_key_val}', 'title': '', 'url': ''}
        decode_end_time = time.perf_counter()
        total_decode_time_batch += (decode_end_time - decode_start_time)
        results_in_original_order[original_pos_idx] = record

    return results_in_original_order, total_get_time_batch, total_decode_time_batch

# --- Warmup query (not timed) ---
warmup_query = "Warmup query for system initialization"
print("Running warmup query (not timed)...")
response = co.embed(texts=[warmup_query], model='multilingual-22-12')
query_embedding = np.array(response.embeddings[0], dtype=np.float32)
query_gpu = cp.asarray(query_embedding, dtype=cp.float32).reshape(1, VECTOR_DIM)
search_params = ivf_pq.SearchParams(n_probes=N_PROBES_SEARCH)
distances_pq, candidates_pq = ivf_pq.search(search_params, index, query_gpu, OVERSAMPLE_K)
candidates_cpu = cp.asnumpy(candidates_pq)[0]
candidate_vecs = get_vectors(candidates_cpu, emb)
scores = np.dot(candidate_vecs, query_embedding)
topk_idx = np.argsort(scores)[-FINAL_K:][::-1]
top_indices = candidates_cpu[topk_idx]
# Warmup metadata fetch (not timed, but exercises the path)
_ = get_metadata_lmdb(top_indices) 
# top_scores = scores[topk_idx] # Already calculated
# for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
#     item = dataset[int(idx)] # Dataset not loaded
#     text = item.get('text', str(item))


# --- User queries (demo) ---
demo_queries = [
    "What is the capital of France?",
    "Explain quantum entanglement.",
    "Best Italian restaurants in New York",
    "What companies did google acquire?",
    "Recent advances in battery technology"
]

# --- Per-query timing storage ---
step_names = [
    'Cohere embedding',
    'Query embedding GPU transfer',
    'IVF-PQ search',
    'Candidate vector fetch',
    'Reranking',
    'LMDB metadata fetch',
    'LMDB raw get',        # New granular timing
    'LMDB decode',         # New granular timing
    'Print results',
    'Total query time'
]
step_times = {name: [] for name in step_names}

try:
    for user_query in demo_queries:
        logging.info(f"\n--- Query: {user_query} ---")
        query_start = time.perf_counter()

        # --- Get embedding from Cohere ---
        step_start_cohere = time.perf_counter()
        print("Getting embedding from Cohere...")
        response = co.embed(texts=[user_query], model='multilingual-22-12')
        query_embedding = np.array(response.embeddings[0], dtype=np.float32)
        step_end_cohere = time.perf_counter()
        elapsed_cohere = step_end_cohere - step_start_cohere
        logging.info(f"Cohere embedding time: {elapsed_cohere:.4f} seconds")
        step_times['Cohere embedding'].append(elapsed_cohere)

        # --- Move query embedding to GPU ---
        step_start_gpu = time.perf_counter()
        query_gpu = cp.asarray(query_embedding, dtype=cp.float32).reshape(1, VECTOR_DIM)
        step_end_gpu = time.perf_counter()
        elapsed_gpu = step_end_gpu - step_start_gpu
        logging.info(f"Query embedding GPU transfer time: {elapsed_gpu:.4f} seconds")
        step_times['Query embedding GPU transfer'].append(elapsed_gpu)

        # --- IVF-PQ search (oversample) ---
        step_start_ivfpq = time.perf_counter()
        print(f"Running IVF-PQ search (top {OVERSAMPLE_K})...")
        search_params = ivf_pq.SearchParams(n_probes=N_PROBES_SEARCH)
        distances_pq, candidates_pq = ivf_pq.search(search_params, index, query_gpu, OVERSAMPLE_K)
        candidates_cpu = cp.asnumpy(candidates_pq)[0]
        step_end_ivfpq = time.perf_counter()
        elapsed_ivfpq = step_end_ivfpq - step_start_ivfpq
        logging.info(f"IVF-PQ search time: {elapsed_ivfpq:.4f} seconds")
        step_times['IVF-PQ search'].append(elapsed_ivfpq)

        # --- Fetch candidate vectors for reranking ---
        step_start_vecfetch = time.perf_counter()
        print("Fetching candidate vectors for reranking...")
        candidate_vecs = get_vectors(candidates_cpu, emb)
        step_end_vecfetch = time.perf_counter()
        elapsed_vecfetch = step_end_vecfetch - step_start_vecfetch
        logging.info(f"Candidate vector fetch time: {elapsed_vecfetch:.4f} seconds")
        step_times['Candidate vector fetch'].append(elapsed_vecfetch)

        # --- Rerank with exact dot product ---
        step_start_rerank = time.perf_counter()
        print("Reranking candidates with exact dot product...")
        scores = np.dot(candidate_vecs, query_embedding)
        topk_idx = np.argsort(scores)[-FINAL_K:][::-1]
        top_indices = candidates_cpu[topk_idx]
        top_scores = scores[topk_idx]
        step_end_rerank = time.perf_counter()
        elapsed_rerank = step_end_rerank - step_start_rerank
        logging.info(f"Reranking time: {elapsed_rerank:.4f} seconds")
        step_times['Reranking'].append(elapsed_rerank)

        # --- Fetch metadata from LMDB ---
        step_start_lmdb_fetch = time.perf_counter()
        print("Fetching metadata from LMDB...")
        metadata_batch, batch_get_time, batch_decode_time = get_metadata_lmdb(top_indices)
        step_end_lmdb_fetch = time.perf_counter()
        elapsed_lmdb_fetch = step_end_lmdb_fetch - step_start_lmdb_fetch
        logging.info(f"LMDB metadata fetch time (overall function): {elapsed_lmdb_fetch:.4f} seconds")
        step_times['LMDB metadata fetch'].append(elapsed_lmdb_fetch)
        logging.info(f"LMDB raw get time (sum for batch): {batch_get_time:.4f} seconds")
        step_times['LMDB raw get'].append(batch_get_time)
        logging.info(f"LMDB decode time (sum for batch): {batch_decode_time:.4f} seconds")
        step_times['LMDB decode'].append(batch_decode_time)
        
        # --- Print results ---
        step_start_print = time.perf_counter()
        print(f"\nTop {FINAL_K} results for query: '{user_query}'\n")
        for rank, (item, score) in enumerate(zip(metadata_batch, top_scores), 1):
            text = item['text'] # Assuming 'text' is always present
            print(f"Rank {rank}: Score {score:.4f}\n{text}\n{'-'*60}")
        step_end_print = time.perf_counter()
        elapsed_print = step_end_print - step_start_print
        logging.info(f"Print results time: {elapsed_print:.4f} seconds")
        step_times['Print results'].append(elapsed_print)

        query_end = time.perf_counter()
        total_query_time = query_end - query_start
        logging.info(f"Total query time: {total_query_time:.4f} seconds")
        step_times['Total query time'].append(total_query_time)

    # --- Log averages ---
    logging.info("\n--- Average timings over all demo queries ---")
    for name in step_names:
        if step_times[name]: # Ensure list is not empty
            avg = sum(step_times[name]) / len(step_times[name])
            logging.info(f"Average {name}: {avg:.4f} seconds")
finally:
    # --- Close LMDB environment ---
    if 'env' in locals() and env is not None:
        env.close()
        print("LMDB environment closed.")

# --- End overall timer ---
overall_end = time.perf_counter()
# Ensure overall_start is defined
if 'overall_start' in locals():
    logging.info(f"Total script execution time: {overall_end - overall_start:.4f} seconds") 