#!/usr/bin/env python3
"""
Tensor Demo 2: Advanced Tensor Operations
Demonstrates advanced tensor manipulations, broadcasting, and optimizations
"""

import numpy as np
import time

def tensor_broadcasting_demo():
    """Demonstrate advanced broadcasting operations"""
    print("=== Advanced Tensor Broadcasting ===\n")
    
    # Create tensors of different shapes
    a = np.random.rand(4, 3, 2)  # 3D tensor
    b = np.random.rand(3, 1)     # 2D tensor
    c = np.random.rand(2)        # 1D tensor
    
    print("1. Tensor shapes:")
    print(f"   A: {a.shape} (3D)")
    print(f"   B: {b.shape} (2D)")
    print(f"   C: {c.shape} (1D)")
    
    # Broadcasting operations
    print("\n2. Broadcasting operations:")
    result1 = a + b  # (4,3,2) + (3,1) -> (4,3,2)
    result2 = a * c  # (4,3,2) * (2,) -> (4,3,2)
    result3 = a + b + c  # Triple broadcasting
    
    print(f"   A + B shape: {result1.shape}")
    print(f"   A * C shape: {result2.shape}")
    print(f"   A + B + C shape: {result3.shape}")
    
    # Advanced indexing
    print("\n3. Advanced indexing:")
    indices = np.array([0, 2])
    selected = a[indices, :, :]  # Select specific slices
    print(f"   Selected slices shape: {selected.shape}")
    
    # Boolean indexing
    mask = a > 0.5
    filtered = a[mask]
    print(f"   Values > 0.5: {len(filtered)} elements")

def einsum_operations():
    """Demonstrate Einstein summation notation"""
    print("\n=== Einstein Summation (einsum) ===\n")
    
    # Create example tensors
    A = np.random.rand(3, 4)
    B = np.random.rand(4, 5)
    C = np.random.rand(3, 5, 2)
    
    print("1. Matrix operations with einsum:")
    
    # Matrix multiplication: A @ B
    result1 = np.einsum('ij,jk->ik', A, B)
    result1_std = A @ B
    print(f"   A @ B using einsum: {result1.shape}")
    print(f"   Results match: {np.allclose(result1, result1_std)}")
    
    # Trace (diagonal sum)
    square_matrix = np.random.rand(5, 5)
    trace = np.einsum('ii->', square_matrix)
    trace_std = np.trace(square_matrix)
    print(f"   Trace using einsum: {trace:.3f}")
    print(f"   Standard trace: {trace_std:.3f}")
    
    # Batch matrix multiplication
    batch_A = np.random.rand(10, 3, 4)
    batch_B = np.random.rand(10, 4, 5)
    batch_result = np.einsum('bij,bjk->bik', batch_A, batch_B)
    print(f"   Batch multiplication: {batch_result.shape}")
    
    # Complex tensor contraction
    tensor_result = np.einsum('ijk,jkl->il', C, np.random.rand(5, 2, 6))
    print(f"   Complex contraction: {tensor_result.shape}")

def tensor_memory_optimization():
    """Demonstrate memory-efficient tensor operations"""
    print("\n=== Memory Optimization Techniques ===\n")
    
    size = 2000
    print(f"Working with {size}x{size} matrices")
    
    # In-place operations
    print("\n1. In-place vs copy operations:")
    A = np.random.rand(size, size)
    
    # Copy operation (uses more memory)
    start_time = time.time()
    B_copy = A + 1.0
    copy_time = time.time() - start_time
    
    # In-place operation (memory efficient)
    A_inplace = A.copy()
    start_time = time.time()
    A_inplace += 1.0
    inplace_time = time.time() - start_time
    
    print(f"   Copy operation: {copy_time:.4f} seconds")
    print(f"   In-place operation: {inplace_time:.4f} seconds")
    print(f"   In-place speedup: {copy_time/inplace_time:.2f}x")
    
    # Memory views
    print("\n2. Memory views and slicing:")
    large_array = np.random.rand(5000, 5000)
    
    # View (no copy)
    start_time = time.time()
    view = large_array[1000:2000, 1000:2000]
    view_time = time.time() - start_time
    
    # Copy
    start_time = time.time()
    copy = large_array[1000:2000, 1000:2000].copy()
    copy_time = time.time() - start_time
    
    print(f"   View creation: {view_time:.6f} seconds")
    print(f"   Copy creation: {copy_time:.6f} seconds")
    print(f"   View is {copy_time/view_time:.0f}x faster")

def vectorized_operations():
    """Demonstrate vectorization for performance"""
    print("\n=== Vectorization Examples ===\n")
    
    n = 1_000_000
    x = np.random.rand(n)
    y = np.random.rand(n)
    
    print(f"Processing {n:,} elements")
    
    # Python loop (slow)
    print("\n1. Python loop:")
    start_time = time.time()
    result_loop = []
    for i in range(len(x)):
        result_loop.append(x[i]**2 + np.sin(y[i]))
    result_loop = np.array(result_loop)
    loop_time = time.time() - start_time
    print(f"   Time: {loop_time:.3f} seconds")
    
    # Vectorized operation (fast)
    print("\n2. Vectorized operation:")
    start_time = time.time()
    result_vectorized = x**2 + np.sin(y)
    vectorized_time = time.time() - start_time
    print(f"   Time: {vectorized_time:.3f} seconds")
    print(f"   Speedup: {loop_time/vectorized_time:.0f}x faster")
    
    # Verify results are the same
    print(f"   Results match: {np.allclose(result_loop, result_vectorized)}")

def tensor_decomposition_demo():
    """Demonstrate tensor decomposition techniques"""
    print("\n=== Tensor Decomposition ===\n")
    
    # Create a low-rank matrix
    rank = 10
    m, n = 500, 300
    U = np.random.rand(m, rank)
    V = np.random.rand(rank, n)
    A = U @ V + 0.1 * np.random.rand(m, n)  # Low-rank + noise
    
    print(f"Original matrix: {A.shape}")
    print(f"True rank: ~{rank}")
    
    # SVD decomposition
    print("\n1. Singular Value Decomposition:")
    start_time = time.time()
    U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
    svd_time = time.time() - start_time
    
    print(f"   SVD time: {svd_time:.3f} seconds")
    print(f"   Singular values (first 15): {s[:15]}")
    
    # Low-rank approximation
    k = 20  # Keep top k components
    A_approx = U_svd[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    # Compression ratio and error
    original_size = A.size
    compressed_size = U_svd[:, :k].size + s[:k].size + Vt[:k, :].size
    compression_ratio = original_size / compressed_size
    reconstruction_error = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
    
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    print(f"   Reconstruction error: {reconstruction_error:.4f}")

def main():
    """Main demo function"""
    tensor_broadcasting_demo()
    einsum_operations()
    tensor_memory_optimization()
    vectorized_operations()
    tensor_decomposition_demo()

if __name__ == "__main__":
    main()