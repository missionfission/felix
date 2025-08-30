#!/usr/bin/env python3
"""
HPC Demo 1: Parallel Matrix Multiplication
Demonstrates multiprocessing for CPU-intensive tasks
"""

import numpy as np
import time
import multiprocessing as mp
from functools import partial

def matrix_multiply_chunk(args):
    """Multiply a chunk of matrix A with matrix B"""
    A_chunk, B, start_row, end_row = args
    return np.dot(A_chunk, B), start_row, end_row

def parallel_matrix_multiply(A, B, num_processes=None):
    """Parallel matrix multiplication using multiprocessing"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    rows_A = A.shape[0]
    chunk_size = rows_A // num_processes
    
    # Create chunks
    chunks = []
    for i in range(num_processes):
        start_row = i * chunk_size
        end_row = start_row + chunk_size if i < num_processes - 1 else rows_A
        A_chunk = A[start_row:end_row]
        chunks.append((A_chunk, B, start_row, end_row))
    
    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        results = pool.map(matrix_multiply_chunk, chunks)
    
    # Combine results
    C = np.zeros((rows_A, B.shape[1]))
    for result, start_row, end_row in results:
        C[start_row:end_row] = result
    
    return C

def demo_parallel_matrix_multiply():
    """Demo function to showcase parallel vs sequential performance"""
    print("=== HPC Demo 1: Parallel Matrix Multiplication ===\n")
    
    # Create large matrices
    size = 1000
    print(f"Creating {size}x{size} matrices...")
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # Sequential multiplication
    print("\n1. Sequential multiplication:")
    start_time = time.time()
    C_sequential = np.dot(A, B)
    sequential_time = time.time() - start_time
    print(f"   Time: {sequential_time:.3f} seconds")
    
    # Parallel multiplication
    num_cores = mp.cpu_count()
    print(f"\n2. Parallel multiplication (using {num_cores} cores):")
    start_time = time.time()
    C_parallel = parallel_matrix_multiply(A, B)
    parallel_time = time.time() - start_time
    print(f"   Time: {parallel_time:.3f} seconds")
    
    # Verify results are the same
    if np.allclose(C_sequential, C_parallel):
        print("   ✓ Results match!")
    else:
        print("   ✗ Results don't match!")
    
    # Performance comparison
    speedup = sequential_time / parallel_time
    print(f"\n3. Performance Analysis:")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Efficiency: {speedup/num_cores:.2f} ({speedup/num_cores*100:.1f}%)")

if __name__ == "__main__":
    demo_parallel_matrix_multiply()