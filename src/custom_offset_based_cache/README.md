# Custom Offset-Based Embedding Cache

This folder contains an experimental offset-based embedding caching system designed as an alternative to numpy memmap for more predictable I/O performance.

## Overview

The system converts standard memmap embeddings into a custom format with explicit file offsets, allowing for more controlled I/O operations and potentially more consistent performance.

## Files

- **`create_offset_embeddings.py`** - Converts existing memmap embeddings to offset-based format
- **`offset_embedding_reader.py`** - Core reader implementation with optimized chunking
- **`compare_embedding_readers.py`** - Performance comparison between memmap and offset approaches

## When This Might Be Useful

- **Network storage** - When page faults are expensive due to network latency
- **Slow storage** - HDDs or other storage where random access is costly
- **Predictable latency** - When you need consistent performance regardless of cache state
- **Memory-constrained systems** - When you can't afford large page cache usage
- **Custom caching strategies** - When you want explicit control over what's cached

## Performance Results

In our testing on fast local SSD storage:
- **Memmap was 55% faster on average** (17.8ms vs 27.6ms)
- **Memmap had better worst-case performance** (80.7ms vs 110.1ms)
- **Offset approach had only 9.9% I/O efficiency** due to scattered access patterns

## Key Insights

1. **OS page cache is quite good** - Modern memory management often outperforms custom solutions
2. **Access patterns matter** - Random access hurts the offset approach's efficiency
3. **Storage speed matters** - Fast SSDs reduce the benefit of avoiding page faults
4. **Always benchmark** - Theoretical optimizations don't always translate to real performance gains

## Usage

```python
# Convert memmap to offset format
python create_offset_embeddings.py

# Use the offset reader
from offset_embedding_reader import OffsetEmbeddingReader

with OffsetEmbeddingReader() as reader:
    embeddings = reader.get_vectors(embedding_ids)

# Compare performance
python compare_embedding_readers.py
```

## Future Improvements

- Better chunking algorithms for scattered access
- Async I/O support
- Compression support
- Multi-file sharding for parallel access
- Smart prefetching based on access patterns

---

*Note: While this approach didn't outperform memmap in our specific use case, it demonstrates important concepts for custom I/O optimization and may be valuable in different scenarios.* 