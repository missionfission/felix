# HPC & Tensor Operations Live Demo

This collection provides comprehensive examples of High Performance Computing (HPC) and tensor operations suitable for live demonstrations. Each example is designed to showcase different aspects of parallel computing, distributed systems, and tensor manipulations.

## 📁 Demo Files Overview

### HPC Examples
- **`1_parallel_matrix_multiply.py`** - Parallel matrix multiplication using multiprocessing
- **`3_mpi_example.py`** - MPI distributed computing examples
- **`4_openmp_simulation.py`** - Thread-based parallel computing patterns
- **`7_distributed_computing.py`** - Advanced distributed computing patterns

### Tensor Examples
- **`2_tensor_operations.py`** - Basic tensor operations with NumPy and PyTorch
- **`5_advanced_tensor_ops.py`** - Advanced tensor manipulations and optimizations
- **`6_gpu_acceleration.py`** - GPU-accelerated computing with CuPy and PyTorch

### Interactive Demo
- **`8_interactive_demo.py`** - Main interactive script to run all demos

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Basic dependencies (required)
pip install numpy scipy

# Optional: For advanced tensor operations
pip install torch torchvision

# Optional: For MPI distributed computing
sudo apt-get install openmpi-bin openmpi-dev  # Linux
pip install mpi4py

# Optional: For GPU acceleration (NVIDIA CUDA required)
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run Interactive Demo

```bash
python 8_interactive_demo.py
```

### 3. Run Individual Demos

```bash
# Parallel matrix multiplication
python 1_parallel_matrix_multiply.py

# Tensor operations
python 2_tensor_operations.py

# MPI distributed computing (requires MPI)
mpirun -n 4 python 3_mpi_example.py

# And so on...
```

## 📊 Demo Descriptions

### 1. Parallel Matrix Multiplication
**File**: `1_parallel_matrix_multiply.py`
- **Concepts**: Multiprocessing, CPU parallelization, performance comparison
- **Demo Time**: ~2-3 minutes
- **Key Points**: 
  - Shows speedup from parallel processing
  - Compares sequential vs parallel performance
  - Demonstrates efficiency calculations

### 2. Basic Tensor Operations
**File**: `2_tensor_operations.py`
- **Concepts**: NumPy arrays, PyTorch tensors, broadcasting, automatic differentiation
- **Demo Time**: ~3-4 minutes
- **Key Points**:
  - Fundamental tensor operations
  - Broadcasting mechanics
  - GPU vs CPU performance comparison

### 3. MPI Distributed Computing
**File**: `3_mpi_example.py`
- **Concepts**: Message passing, distributed memory, process communication
- **Demo Time**: ~4-5 minutes
- **Requirements**: MPI installation and mpi4py
- **Key Points**:
  - Inter-process communication
  - Distributed matrix operations
  - Scalability across multiple nodes

### 4. OpenMP-style Simulation
**File**: `4_openmp_simulation.py`
- **Concepts**: Thread-based parallelism, Monte Carlo methods, particle simulation
- **Demo Time**: ~3-4 minutes
- **Key Points**:
  - Shared memory parallelism
  - Embarrassingly parallel problems
  - Thread synchronization

### 5. Advanced Tensor Operations
**File**: `5_advanced_tensor_ops.py`
- **Concepts**: Broadcasting, einsum, memory optimization, vectorization
- **Demo Time**: ~4-5 minutes
- **Key Points**:
  - Einstein summation notation
  - Memory-efficient operations
  - Vectorization benefits

### 6. GPU Acceleration
**File**: `6_gpu_acceleration.py`
- **Concepts**: CUDA, GPU computing, memory transfer overhead
- **Demo Time**: ~3-4 minutes
- **Requirements**: NVIDIA GPU with CUDA
- **Key Points**:
  - Massive parallel processing
  - Memory transfer considerations
  - When GPU acceleration helps

### 7. Distributed Computing Patterns
**File**: `7_distributed_computing.py`
- **Concepts**: MapReduce, load balancing, work distribution
- **Demo Time**: ~5-6 minutes
- **Key Points**:
  - Fault tolerance patterns
  - Dynamic load balancing
  - Big data processing paradigms

## 🎯 Live Demo Tips

### For Presenters

1. **Start with Interactive Demo**: Use `8_interactive_demo.py` for guided experience
2. **Check Dependencies**: Run dependency check first to see what's available
3. **Timing**: Each demo runs 2-6 minutes - perfect for live presentations
4. **Audience Interaction**: Ask audience to predict performance improvements
5. **Visual Impact**: Large speedup numbers create good "wow" moments

### Demo Flow Suggestions

**Option 1: Quick Overview (15 minutes)**
1. Run interactive demo, select demos 1, 2, 5
2. Focus on performance comparisons and speedup numbers

**Option 2: Deep Dive (30 minutes)**
1. Start with basic concepts (demo 2)
2. Show parallel computing benefits (demo 1)
3. Demonstrate advanced techniques (demos 5, 6)
4. Conclude with distributed patterns (demo 7)

**Option 3: Specific Focus**
- **HPC Focus**: Demos 1, 3, 4, 7
- **Tensor Focus**: Demos 2, 5, 6
- **GPU Focus**: Demos 6, 2 (PyTorch sections)

## 🔧 Troubleshooting

### Common Issues

1. **MPI not working**: 
   ```bash
   sudo apt-get install openmpi-bin openmpi-dev
   pip install mpi4py
   ```

2. **GPU demos not working**:
   - Check NVIDIA drivers: `nvidia-smi`
   - Install CUDA toolkit
   - Install appropriate CuPy version

3. **Performance not showing speedup**:
   - Ensure sufficient problem size
   - Check system load
   - Verify multiple CPU cores available

### Performance Notes

- **Matrix sizes**: Demos use large matrices (1000x1000+) to show meaningful speedups
- **CPU cores**: Performance scales with available CPU cores
- **Memory**: Some demos require several GB of RAM
- **GPU memory**: GPU demos may need 2GB+ GPU memory

## 📈 Expected Performance Results

### Typical Speedups (4-core system)
- **Parallel matrix multiply**: 2-3x speedup
- **Monte Carlo π**: 3-4x speedup  
- **GPU acceleration**: 10-100x speedup (problem size dependent)
- **Vectorization**: 50-200x speedup over Python loops

### Demo Timing
- Each individual demo: 2-6 minutes
- Full interactive session: 20-30 minutes
- Quick highlights: 10-15 minutes

## 🎓 Educational Value

### HPC Concepts Demonstrated
- **Parallelization strategies**: Data vs task parallelism
- **Performance analysis**: Speedup, efficiency, scalability
- **Memory management**: Cache efficiency, memory bandwidth
- **Communication patterns**: Shared vs distributed memory

### Tensor Concepts Demonstrated
- **Broadcasting rules**: How operations work across dimensions
- **Memory layout**: Row-major vs column-major, views vs copies
- **Optimization techniques**: Vectorization, in-place operations
- **Hardware acceleration**: CPU vs GPU trade-offs

## 🔗 Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MPI Tutorial](https://mpitutorial.com/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

---

**Ready to demo?** Run `python 8_interactive_demo.py` and select your demos!