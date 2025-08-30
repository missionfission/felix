#!/usr/bin/env python3
"""
Tensor Demo 1: Basic Tensor Operations
Demonstrates fundamental tensor operations using NumPy and PyTorch
"""

import numpy as np
import time

def numpy_tensor_demo():
    """Demonstrate tensor operations using NumPy"""
    print("=== NumPy Tensor Operations ===\n")
    
    # Create tensors
    print("1. Creating tensors:")
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8], [9, 10], [11, 12]])
    print(f"   Tensor A (2x3):\n{a}")
    print(f"   Tensor B (3x2):\n{b}")
    
    # Basic operations
    print("\n2. Basic operations:")
    print(f"   A + 10:\n{a + 10}")
    print(f"   A * 2:\n{a * 2}")
    print(f"   A squared:\n{a**2}")
    
    # Matrix multiplication
    print("\n3. Matrix multiplication:")
    c = np.dot(a, b)
    print(f"   A @ B (2x2):\n{c}")
    
    # Tensor reshaping
    print("\n4. Reshaping:")
    a_reshaped = a.reshape(3, 2)
    print(f"   A reshaped to 3x2:\n{a_reshaped}")
    
    # Broadcasting
    print("\n5. Broadcasting:")
    vector = np.array([1, 2, 3])
    broadcasted = a + vector
    print(f"   A + [1,2,3] (broadcasting):\n{broadcasted}")
    
    # Aggregations
    print("\n6. Aggregations:")
    print(f"   Sum: {np.sum(a)}")
    print(f"   Mean: {np.mean(a):.2f}")
    print(f"   Max: {np.max(a)}")
    print(f"   Std: {np.std(a):.2f}")

def pytorch_tensor_demo():
    """Demonstrate tensor operations using PyTorch (if available)"""
    try:
        import torch
        print("\n=== PyTorch Tensor Operations ===\n")
        
        # Create tensors
        print("1. Creating PyTorch tensors:")
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        print(f"   Tensor A:\n{a}")
        print(f"   Tensor B:\n{b}")
        
        # GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        # Move to device
        a = a.to(device)
        b = b.to(device)
        
        # Operations
        print("\n2. Tensor operations:")
        print(f"   A + 10:\n{a + 10}")
        print(f"   A * 2:\n{a * 2}")
        
        # Matrix multiplication
        print("\n3. Matrix multiplication:")
        c = torch.mm(a, b)
        print(f"   A @ B:\n{c}")
        
        # Gradients (for deep learning)
        print("\n4. Automatic differentiation:")
        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        y = x**2 + 3*x + 1
        y.backward(torch.ones_like(y))
        print(f"   x = {x.data}")
        print(f"   y = x² + 3x + 1 = {y.data}")
        print(f"   dy/dx = 2x + 3 = {x.grad}")
        
    except ImportError:
        print("\n=== PyTorch not available ===")
        print("Install with: pip install torch")

def tensor_performance_comparison():
    """Compare performance of different tensor operations"""
    print("\n=== Performance Comparison ===\n")
    
    # Large matrix operations
    size = 2000
    print(f"Creating {size}x{size} matrices for performance test...")
    
    # NumPy
    np.random.seed(42)
    A_np = np.random.rand(size, size)
    B_np = np.random.rand(size, size)
    
    print("\n1. NumPy performance:")
    start_time = time.time()
    C_np = np.dot(A_np, B_np)
    numpy_time = time.time() - start_time
    print(f"   Matrix multiplication: {numpy_time:.3f} seconds")
    
    # PyTorch (if available)
    try:
        import torch
        A_torch = torch.from_numpy(A_np).float()
        B_torch = torch.from_numpy(B_np).float()
        
        print("\n2. PyTorch CPU performance:")
        start_time = time.time()
        C_torch = torch.mm(A_torch, B_torch)
        pytorch_time = time.time() - start_time
        print(f"   Matrix multiplication: {pytorch_time:.3f} seconds")
        
        # GPU performance (if available)
        if torch.cuda.is_available():
            A_gpu = A_torch.cuda()
            B_gpu = B_torch.cuda()
            
            print("\n3. PyTorch GPU performance:")
            # Warm up GPU
            _ = torch.mm(A_gpu, B_gpu)
            torch.cuda.synchronize()
            
            start_time = time.time()
            C_gpu = torch.mm(A_gpu, B_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            print(f"   Matrix multiplication: {gpu_time:.3f} seconds")
            print(f"   GPU speedup: {pytorch_time/gpu_time:.2f}x over CPU")
        
    except ImportError:
        print("\n2. PyTorch not available for comparison")

def main():
    """Main demo function"""
    numpy_tensor_demo()
    pytorch_tensor_demo()
    tensor_performance_comparison()

if __name__ == "__main__":
    main()