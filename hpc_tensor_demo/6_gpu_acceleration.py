#!/usr/bin/env python3
"""
Tensor Demo 3: GPU Acceleration
Demonstrates GPU-accelerated tensor operations using CuPy and PyTorch
"""

import numpy as np
import time

def cupy_demo():
    """Demonstrate GPU acceleration with CuPy"""
    try:
        import cupy as cp
        print("=== CuPy GPU Acceleration Demo ===\n")
        
        # Check GPU availability
        print(f"GPU device: {cp.cuda.Device()}")
        print(f"GPU memory: {cp.cuda.MemoryInfo()}")
        
        # Create large arrays
        size = 5000
        print(f"\nCreating {size}x{size} matrices...")
        
        # CPU arrays
        A_cpu = np.random.rand(size, size).astype(np.float32)
        B_cpu = np.random.rand(size, size).astype(np.float32)
        
        # GPU arrays
        A_gpu = cp.asarray(A_cpu)
        B_gpu = cp.asarray(B_cpu)
        
        # CPU computation
        print("\n1. CPU computation (NumPy):")
        start_time = time.time()
        C_cpu = np.dot(A_cpu, B_cpu)
        cpu_time = time.time() - start_time
        print(f"   Time: {cpu_time:.3f} seconds")
        
        # GPU computation
        print("\n2. GPU computation (CuPy):")
        start_time = time.time()
        C_gpu = cp.dot(A_gpu, B_gpu)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        print(f"   Time: {gpu_time:.3f} seconds")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Verify results
        C_gpu_cpu = cp.asnumpy(C_gpu)
        if np.allclose(C_cpu, C_gpu_cpu, rtol=1e-5):
            print("   ✓ Results match!")
        
        # Element-wise operations
        print("\n3. Element-wise operations:")
        start_time = time.time()
        result_cpu = np.sin(A_cpu) * np.exp(B_cpu) + np.sqrt(A_cpu * B_cpu)
        cpu_elem_time = time.time() - start_time
        
        start_time = time.time()
        result_gpu = cp.sin(A_gpu) * cp.exp(B_gpu) + cp.sqrt(A_gpu * B_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_elem_time = time.time() - start_time
        
        print(f"   CPU time: {cpu_elem_time:.3f} seconds")
        print(f"   GPU time: {gpu_elem_time:.3f} seconds")
        print(f"   GPU speedup: {cpu_elem_time/gpu_elem_time:.2f}x")
        
    except ImportError:
        print("=== CuPy not available ===")
        print("Install with: pip install cupy-cuda11x (or appropriate CUDA version)")
        print("Requires NVIDIA GPU with CUDA support")

def pytorch_gpu_demo():
    """Demonstrate GPU acceleration with PyTorch"""
    try:
        import torch
        print("\n=== PyTorch GPU Acceleration Demo ===\n")
        
        # Check GPU availability
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"GPU device: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            device = torch.device('cpu')
            print("GPU not available, using CPU")
        
        # Create tensors
        size = 4000
        print(f"\nCreating {size}x{size} tensors...")
        
        A_cpu = torch.randn(size, size)
        B_cpu = torch.randn(size, size)
        
        # CPU computation
        print("\n1. CPU computation:")
        start_time = time.time()
        C_cpu = torch.mm(A_cpu, B_cpu)
        cpu_time = time.time() - start_time
        print(f"   Time: {cpu_time:.3f} seconds")
        
        if torch.cuda.is_available():
            # GPU computation
            A_gpu = A_cpu.to(device)
            B_gpu = B_cpu.to(device)
            
            print("\n2. GPU computation:")
            # Warm up
            _ = torch.mm(A_gpu, B_gpu)
            torch.cuda.synchronize()
            
            start_time = time.time()
            C_gpu = torch.mm(A_gpu, B_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"   Time: {gpu_time:.3f} seconds")
            print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
            
            # Memory transfer overhead
            print("\n3. Memory transfer analysis:")
            start_time = time.time()
            A_transfer = A_cpu.to(device)
            transfer_time = time.time() - start_time
            print(f"   CPU->GPU transfer: {transfer_time:.3f} seconds")
            
            start_time = time.time()
            A_back = A_gpu.cpu()
            back_transfer_time = time.time() - start_time
            print(f"   GPU->CPU transfer: {back_transfer_time:.3f} seconds")
            
            total_with_transfer = transfer_time + gpu_time + back_transfer_time
            print(f"   Total with transfers: {total_with_transfer:.3f} seconds")
            print(f"   Effective speedup: {cpu_time/total_with_transfer:.2f}x")
        
    except ImportError:
        print("=== PyTorch not available ===")
        print("Install with: pip install torch")

def neural_network_tensor_ops():
    """Demonstrate tensor operations in neural network context"""
    try:
        import torch
        import torch.nn.functional as F
        print("\n=== Neural Network Tensor Operations ===\n")
        
        # Simulate a batch of data
        batch_size = 64
        input_size = 784  # 28x28 images
        hidden_size = 256
        output_size = 10
        
        print(f"Batch size: {batch_size}")
        print(f"Input size: {input_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Output size: {output_size}")
        
        # Create random data and weights
        X = torch.randn(batch_size, input_size)
        W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        b1 = torch.randn(hidden_size, requires_grad=True)
        W2 = torch.randn(hidden_size, output_size, requires_grad=True)
        b2 = torch.randn(output_size, requires_grad=True)
        
        print("\n1. Forward pass:")
        start_time = time.time()
        
        # Forward pass
        h1 = torch.mm(X, W1) + b1  # Linear layer 1
        h1_relu = F.relu(h1)       # ReLU activation
        output = torch.mm(h1_relu, W2) + b2  # Linear layer 2
        probabilities = F.softmax(output, dim=1)  # Softmax
        
        forward_time = time.time() - start_time
        print(f"   Forward pass time: {forward_time:.4f} seconds")
        print(f"   Output shape: {output.shape}")
        print(f"   Probability sum (should be ~1): {probabilities.sum(dim=1).mean():.3f}")
        
        # Backward pass
        print("\n2. Backward pass (automatic differentiation):")
        target = torch.randint(0, output_size, (batch_size,))
        loss = F.cross_entropy(output, target)
        
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Backward pass time: {backward_time:.4f} seconds")
        print(f"   W1 gradient shape: {W1.grad.shape}")
        print(f"   W1 gradient norm: {W1.grad.norm().item():.4f}")
        
    except ImportError:
        print("=== PyTorch not available ===")
        print("Install with: pip install torch")

def tensor_fft_demo():
    """Demonstrate Fast Fourier Transform operations"""
    print("\n=== Tensor FFT Operations ===\n")
    
    # Create a signal with multiple frequencies
    t = np.linspace(0, 1, 1000, endpoint=False)
    signal = (np.sin(2 * np.pi * 5 * t) +  # 5 Hz
              0.5 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz
              0.3 * np.sin(2 * np.pi * 20 * t))   # 20 Hz
    
    # Add noise
    noise = 0.1 * np.random.randn(len(t))
    noisy_signal = signal + noise
    
    print("1. Signal analysis:")
    print(f"   Signal length: {len(signal)}")
    print(f"   Sampling rate: {len(signal)} Hz")
    print(f"   Signal contains: 5Hz, 10Hz, 20Hz components")
    
    # FFT
    print("\n2. Fast Fourier Transform:")
    start_time = time.time()
    fft_result = np.fft.fft(noisy_signal)
    fft_time = time.time() - start_time
    
    frequencies = np.fft.fftfreq(len(signal), 1/len(signal))
    magnitude = np.abs(fft_result)
    
    print(f"   FFT time: {fft_time:.4f} seconds")
    
    # Find dominant frequencies
    positive_freq_mask = frequencies > 0
    positive_freqs = frequencies[positive_freq_mask]
    positive_mags = magnitude[positive_freq_mask]
    
    # Get top 5 frequencies
    top_indices = np.argsort(positive_mags)[-5:][::-1]
    print(f"   Top 5 frequencies:")
    for i, idx in enumerate(top_indices):
        freq = positive_freqs[idx]
        mag = positive_mags[idx]
        print(f"     {i+1}. {freq:.1f} Hz (magnitude: {mag:.0f})")
    
    # 2D FFT example
    print("\n3. 2D FFT (image processing):")
    image_size = 512
    # Create a synthetic image with patterns
    x = np.linspace(0, 1, image_size)
    y = np.linspace(0, 1, image_size)
    X, Y = np.meshgrid(x, y)
    image = np.sin(10 * np.pi * X) * np.cos(8 * np.pi * Y)
    
    start_time = time.time()
    fft_2d = np.fft.fft2(image)
    fft_2d_time = time.time() - start_time
    
    print(f"   Image size: {image.shape}")
    print(f"   2D FFT time: {fft_2d_time:.4f} seconds")
    print(f"   Frequency domain shape: {fft_2d.shape}")

def main():
    """Main demo function"""
    tensor_broadcasting_demo()
    einsum_operations()
    tensor_memory_optimization()
    vectorized_operations()
    neural_network_tensor_ops()
    tensor_fft_demo()
    cupy_demo()
    pytorch_gpu_demo()

if __name__ == "__main__":
    main()