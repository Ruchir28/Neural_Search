import struct
import numpy as np
from typing import List, Optional
from datasets import load_dataset

def load_list_sizes(filepath: str = "lists_info/sizes.bin") -> Optional[List[int]]:
    sizes = []
    uint32_size = 4
    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(uint32_size)
                if not chunk:
                    break
                if len(chunk) < uint32_size:
                    print(f"Warning: Partial chunk at end of '{filepath}'.")
                    break
                size_tuple = struct.unpack('<I', chunk)
                sizes.append(size_tuple[0])
        return sizes
    except Exception as e:
        print(f"Error reading '{filepath}': {e}")
        return None

def load_list_ids(list_sizes: List[int], filepath: str = "lists_info/ids.bin") -> Optional[List[List[int]]]:
    lists = []
    int64_size = 8
    try:
        with open(filepath, 'rb') as f:
            for size in list_sizes:
                if size > 0:
                    chunk = f.read(size * int64_size)
                    if len(chunk) < size * int64_size:
                        print(f"Error: Expected {size * int64_size} bytes, got {len(chunk)} bytes.")
                        return None
                    format_string = f'<{size}q'
                    ids = struct.unpack(format_string, chunk)
                    lists.append(list(ids))
                else:
                    lists.append([])
        return lists
    except Exception as e:
        print(f"Error reading '{filepath}': {e}")
        return None

def create_ivf_pq_layout():
    sizes = load_list_sizes()
    ids = load_list_ids(sizes)
    if ids is None:
        raise ValueError("Failed to load list IDs")
    prefix = np.insert(np.cumsum(sizes), 0, 0)
    row2list = np.empty(prefix[-1], dtype=np.uint32)
    for list_idx, list_ids in enumerate(ids):
        row2list[list_ids] = list_idx
    np.savez("lists_info/ivf_pq_layout.npz", row2list=row2list, prefix=prefix, sizes=sizes)

def create_initial_embeddings_memorymap(
    dataset_name: str = "Cohere/wikipedia-22-12-en-embeddings",
    cache_dir: str = "dataset_cache",
    output_path: str = "lists_info/embeddings_all.memmap",
    vector_dim: int = 768
):
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    dataset = dataset.with_format("numpy")
    total_embeddings = len(dataset)
    print(f"Total embeddings to write: {total_embeddings}")
    mmap = np.memmap(output_path, dtype=np.float16, mode="w+", shape=(total_embeddings, vector_dim))
    batch_size = 100_000
    for start in range(0, total_embeddings, batch_size):
        end = min(start + batch_size, total_embeddings)
        batch = dataset[start:end]["emb"]
        mmap[start:end] = np.array(batch, dtype=np.float16)
        print(f"Wrote embeddings {start} to {end-1}")
    mmap.flush()
    print(f"All embeddings memory-mapped file created at {output_path}")

def create_embedding_memorymap_wrt_list_ids(
    embeddings_path: str = "lists_info/embeddings_all.memmap",
    output_path: str = "lists_info/embeddings_listwise.memmap",
    layout_path: str = "lists_info/ivf_pq_layout.npz",
    id_to_sorted_row_path: str = "lists_info/vec_id_to_sorted_row.bin.npy"
):
    layout = np.load(layout_path)
    row2list = layout["row2list"]
    prefix = layout["prefix"]
    perm = np.argsort(row2list)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    np.save(id_to_sorted_row_path, inv_perm)

    emb_in = np.memmap(embeddings_path, dtype=np.float16, mode="r", shape=(len(row2list), 768))
    emb_out = np.memmap(output_path, dtype=np.float16, mode="w+", shape=emb_in.shape)
    CHUNK_SIZE = 1_000_000
    for i in range(0, len(perm), CHUNK_SIZE):
        j = min(i + CHUNK_SIZE, len(perm))
        emb_out[i:j] = emb_in[perm[i:j]]
    emb_out.flush()
    print(f"Listwise embedding memory-mapped file created at {output_path}")

    # Save sorted layout for fast block access
    sizes_sorted = np.bincount(row2list[perm])
    prefix_sorted = np.insert(np.cumsum(sizes_sorted), 0, 0)
    np.savez("lists_info/layout_sorted.npz", sizes=sizes_sorted, prefix=prefix_sorted)

if __name__ == "__main__":
    create_ivf_pq_layout()
    create_initial_embeddings_memorymap()
    create_embedding_memorymap_wrt_list_ids() 