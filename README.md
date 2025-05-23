# High-Performance Neural Search Engine

A production-ready semantic search system built with NVIDIA cuVS, featuring advanced vector quantization and optimized memory access patterns for sub-100ms query latency on 35M high-dimensional embeddings.

## 🎯 Project Overview

This project implements a complete end-to-end neural search pipeline capable of handling large-scale semantic search across 35 million 768-dimensional Wikipedia embeddings. The system achieves **13x storage compression** and **<100ms average query latency** through advanced indexing techniques and memory optimization.

## 🚀 Key Technical Achievements

### **1. Vector Indexing & Compression**
- **Problem**: 35M vectors × 768 dims × 2 bytes = 53.76 GB storage requirement
- **Solution**: Implemented IVF-PQ (Inverted File with Product Quantization) using NVIDIA cuVS
- **Result**: Compressed to 3.36 GB (96 bytes/vector) - **93.7% storage reduction**

**Technical Details:**
- 32,768 IVF clusters for coarse quantization
- 96 PQ sub-quantizers (8 dimensions each, 8 bits per code)
- Trained on 2M representative vectors, populated with remaining 33M in 500K batches

### **2. Recall Optimization Through Oversampling**
- **Problem**: Direct PQ search yielded only 60% recall@10
- **Solution**: Implemented two-stage retrieval with oversampling + reranking
- **Result**: Achieved **89% recall@10** while maintaining performance

**Implementation:**
```
Stage 1: Retrieve 1000 candidates using PQ approximation
Stage 2: Exact cosine similarity reranking on original embeddings
```

### **3. Memory Access Pattern Optimization**
- **Problem**: Random memory access for 1000 vectors took 800ms (unacceptable)
- **Solution**: List-wise memory layout optimization leveraging IVF structure
- **Result**: **1-10ms average retrieval**, 100ms worst-case (**80x improvement**)

**Key Insight:** With n_probes=40, results come from max 40 lists. Arranging embeddings by list enables sequential memory access patterns.

### **4. Production-Ready API Server**
- FastAPI-based REST endpoint with comprehensive error handling
- Async request processing with connection pooling
- Integrated monitoring and logging
- Memory-mapped file handling for efficient resource utilization

## 📊 Performance Metrics

| Metric | Value | Baseline Comparison |
|--------|-------|-------------------|
| **Storage Compression** | 13.0x | 53.76 GB → 3.36 GB |
| **Query Latency (avg)** | <100ms | - |
| **Recall@10** | 89% | 60% (direct PQ) |
| **Memory Access Speed** | 1-10ms | 800ms (random access) |

## 🏗️ System Architecture

```
Query → Cohere API → GPU Index Search → Memory-Optimized Retrieval → Metadata Lookup → Results
  70ms      20ms           1-10ms              1-5ms           <100ms total
```

### **Core Components:**

1. **Index Training Pipeline** (`src/train_index.py`)
   - Efficient data loading with multiprocessing
   - GPU memory monitoring and optimization
   - Configurable training parameters

2. **Optimized Search Engine** (`src/search_index.py`)
   - Two-stage retrieval implementation
   - List-wise memory access patterns
   - Batch processing for metadata retrieval

3. **Production Server** (`server.py`)
   - FastAPI with async processing
   - LMDB for metadata storage
   - Comprehensive error handling and logging

4. **Memory Management** (`src/utils/`)
   - Custom offset-based caching
   - GPU memory monitoring
   - Validation utilities

## 🔧 Technical Implementation Details

### **Vector Quantization Strategy**
- **Coarse Quantization**: 32,768 clusters using k-means
- **Fine Quantization**: 96 sub-quantizers with 256 centroids each
- **Distance Metric**: Inner product (optimized for normalized vectors)

### **Memory Layout Optimization**
```python
# Original: Random access across 35M vectors
embeddings[random_ids]  # 800ms for 1000 vectors

# Optimized: List-wise sequential access
for list_id in active_lists:
    embeddings[list_start:list_end][local_offsets]  # 1-10ms total
```

### **Metadata Management**
- LMDB for fast key-value retrieval
- Sorted access patterns for cache efficiency
- Binary encoding for space optimization

## 🛠️ Technology Stack

- **Vector Processing**: NVIDIA cuVS, CuPy, NumPy
- **API Framework**: FastAPI, Uvicorn
- **Data Storage**: LMDB, Memory-mapped files
- **ML Pipeline**: Hugging Face Datasets, Cohere API
- **Monitoring**: Custom GPU memory tracking, structured logging

## 📈 Scalability Considerations

- **Horizontal Scaling**: Stateless API design enables load balancing
- **Memory Efficiency**: Memory-mapped files reduce RAM requirements
- **GPU Utilization**: Batch processing maximizes throughput
- **Cache Optimization**: List-wise layout improves OS page cache hits

## 🚀 Quick Start

### **Prerequisites**
- NVIDIA GPU with CUDA 12.1+
- Python 3.8+
- 8GB+ GPU memory

### **Installation**
```bash
# Clone repository
git clone <repository-url>
cd neural-search-engine

# Install dependencies
pip install -r requirements.txt


# Train index (requires ~4GB GPU memory)


## 🔍 Future Enhancements

1. **Multi-GPU Support**: Distribute index across multiple GPUs
2. **Dynamic Updates**: Implement incremental index updates
3. **Query Optimization**: Adaptive n_probes based on query characteristics
4. **Caching Layer**: Redis for frequently accessed results

## 📝 Key Learnings & Problem-Solving

1. **Memory Access Patterns Matter**: Achieved 80x speedup through data layout optimization
2. **Quantization Trade-offs**: Balanced compression vs. accuracy through two-stage retrieval
3. **System Integration**: Built complete pipeline from training to production deployment
4. **Performance Monitoring**: Implemented comprehensive logging for optimization insights

