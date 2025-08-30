#!/usr/bin/env python3
"""
Benchmark Suite for HPC and Tensor Operations
Comprehensive benchmarking tool for performance analysis
"""

import numpy as np
import time
import multiprocessing as mp
import sys
import psutil
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class BenchmarkResult:
    """Structure to hold benchmark results"""
    name: str
    execution_time: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    additional_metrics: Dict[str, Any]

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.system_info = self._get_system_info()
    
    def _get_system_info(self):
        """Gather system information"""
        return {
            'cpu_count': mp.cpu_count(),
            'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
            'python_version': sys.version,
            'numpy_version': np.__version__
        }
    
    def _monitor_resources(self, func, *args, **kwargs):
        """Monitor resource usage during function execution"""
        process = psutil.Process()
        
        # Initial measurements
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        start_time = time.time()
        cpu_percent_start = process.cpu_percent()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Final measurements
        end_time = time.time()
        final_memory = process.memory_info().rss / (1024**2)  # MB
        cpu_percent_end = process.cpu_percent()
        
        execution_time = end_time - start_time
        memory_usage = final_memory - initial_memory
        cpu_usage = (cpu_percent_start + cpu_percent_end) / 2
        
        return result, execution_time, memory_usage, cpu_usage
    
    def benchmark_matrix_operations(self):
        """Benchmark various matrix operations"""
        print("=== Matrix Operations Benchmark ===\n")
        
        sizes = [500, 1000, 2000, 3000]
        
        for size in sizes:
            print(f"Matrix size: {size}x{size}")
            
            # Create matrices
            A = np.random.rand(size, size).astype(np.float32)
            B = np.random.rand(size, size).astype(np.float32)
            
            # Matrix multiplication
            def matmul_test():
                return np.dot(A, B)
            
            _, exec_time, memory, cpu = self._monitor_resources(matmul_test)
            throughput = (2 * size**3) / (exec_time * 1e9)  # GFLOPS
            
            result = BenchmarkResult(
                name=f"MatMul_{size}x{size}",
                execution_time=exec_time,
                throughput=throughput,
                memory_usage=memory,
                cpu_usage=cpu,
                additional_metrics={'matrix_size': size}
            )
            
            self.results.append(result)
            print(f"  Time: {exec_time:.3f}s, Throughput: {throughput:.2f} GFLOPS")
        
        print()
    
    def benchmark_parallel_scaling(self):
        """Benchmark parallel scaling with different core counts"""
        print("=== Parallel Scaling Benchmark ===\n")
        
        def parallel_computation(num_cores):
            """CPU-intensive computation using specified number of cores"""
            def worker_task(data_chunk):
                return np.sum(np.sin(data_chunk) * np.cos(data_chunk))
            
            # Create large dataset
            data = np.random.rand(10_000_000)
            chunk_size = len(data) // num_cores
            chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_cores)]
            
            with mp.Pool(num_cores) as pool:
                results = pool.map(worker_task, chunks)
            
            return sum(results)
        
        max_cores = min(mp.cpu_count(), 8)  # Test up to 8 cores
        
        for cores in range(1, max_cores + 1):
            _, exec_time, memory, cpu = self._monitor_resources(parallel_computation, cores)
            
            # Calculate efficiency
            if cores == 1:
                baseline_time = exec_time
                efficiency = 100.0
            else:
                speedup = baseline_time / exec_time
                efficiency = (speedup / cores) * 100
            
            result = BenchmarkResult(
                name=f"Parallel_{cores}_cores",
                execution_time=exec_time,
                throughput=cores / exec_time,  # Tasks per second
                memory_usage=memory,
                cpu_usage=cpu,
                additional_metrics={'cores': cores, 'efficiency': efficiency}
            )
            
            self.results.append(result)
            print(f"  {cores} cores: {exec_time:.3f}s (efficiency: {efficiency:.1f}%)")
        
        print()
    
    def benchmark_memory_patterns(self):
        """Benchmark different memory access patterns"""
        print("=== Memory Access Pattern Benchmark ===\n")
        
        size = 2000
        matrix = np.random.rand(size, size)
        
        # Row-major access (cache-friendly)
        def row_major_access():
            total = 0
            for i in range(size):
                for j in range(size):
                    total += matrix[i, j]
            return total
        
        # Column-major access (cache-unfriendly)
        def column_major_access():
            total = 0
            for j in range(size):
                for i in range(size):
                    total += matrix[i, j]
            return total
        
        # Vectorized access
        def vectorized_access():
            return np.sum(matrix)
        
        patterns = [
            ("Row-major", row_major_access),
            ("Column-major", column_major_access),
            ("Vectorized", vectorized_access)
        ]
        
        for name, func in patterns:
            _, exec_time, memory, cpu = self._monitor_resources(func)
            throughput = (size * size) / (exec_time * 1e6)  # Million elements/sec
            
            result = BenchmarkResult(
                name=f"Memory_{name}",
                execution_time=exec_time,
                throughput=throughput,
                memory_usage=memory,
                cpu_usage=cpu,
                additional_metrics={'pattern': name}
            )
            
            self.results.append(result)
            print(f"  {name}: {exec_time:.3f}s ({throughput:.2f} Melem/s)")
        
        print()
    
    def benchmark_tensor_operations(self):
        """Benchmark various tensor operations"""
        print("=== Tensor Operations Benchmark ===\n")
        
        # Test different tensor sizes
        sizes = [(1000, 1000), (2000, 2000), (5000, 1000), (1000, 5000)]
        
        operations = [
            ("Addition", lambda A, B: A + B),
            ("Multiplication", lambda A, B: A * B),
            ("Matrix Multiply", lambda A, B: np.dot(A, B) if A.shape[1] == B.shape[0] else A @ B.T),
            ("Element-wise Sin", lambda A, B: np.sin(A)),
            ("SVD", lambda A, B: np.linalg.svd(A, full_matrices=False)),
        ]
        
        for rows, cols in sizes:
            print(f"Tensor size: {rows}x{cols}")
            A = np.random.rand(rows, cols).astype(np.float32)
            B = np.random.rand(rows, cols).astype(np.float32)
            
            for op_name, op_func in operations:
                try:
                    _, exec_time, memory, cpu = self._monitor_resources(op_func, A, B)
                    throughput = (rows * cols) / (exec_time * 1e6)  # Melem/s
                    
                    result = BenchmarkResult(
                        name=f"{op_name}_{rows}x{cols}",
                        execution_time=exec_time,
                        throughput=throughput,
                        memory_usage=memory,
                        cpu_usage=cpu,
                        additional_metrics={'operation': op_name, 'shape': (rows, cols)}
                    )
                    
                    self.results.append(result)
                    print(f"    {op_name}: {exec_time:.4f}s")
                    
                except Exception as e:
                    print(f"    {op_name}: Error - {e}")
            
            print()
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("=" * 60)
        print("           BENCHMARK REPORT")
        print("=" * 60)
        
        # System information
        print(f"\nSystem Information:")
        print(f"  CPU Cores: {self.system_info['cpu_count']}")
        print(f"  Total Memory: {self.system_info['memory_total']} GB")
        print(f"  Python Version: {self.system_info['python_version'].split()[0]}")
        print(f"  NumPy Version: {self.system_info['numpy_version']}")
        
        # Performance summary
        print(f"\nPerformance Summary:")
        print(f"  Total benchmarks run: {len(self.results)}")
        
        if self.results:
            avg_time = np.mean([r.execution_time for r in self.results])
            max_throughput = max([r.throughput for r in self.results])
            
            print(f"  Average execution time: {avg_time:.3f} seconds")
            print(f"  Peak throughput: {max_throughput:.2f} operations/second")
        
        # Detailed results
        print(f"\nDetailed Results:")
        print(f"{'Benchmark':<25} {'Time (s)':<10} {'Throughput':<12} {'Memory (MB)':<12}")
        print("-" * 65)
        
        for result in self.results:
            print(f"{result.name:<25} {result.execution_time:<10.3f} "
                  f"{result.throughput:<12.2f} {result.memory_usage:<12.1f}")
        
        # Performance insights
        self._generate_insights()
    
    def _generate_insights(self):
        """Generate performance insights"""
        print(f"\nPerformance Insights:")
        
        # Find fastest and slowest operations
        if len(self.results) >= 2:
            fastest = min(self.results, key=lambda x: x.execution_time)
            slowest = max(self.results, key=lambda x: x.execution_time)
            
            print(f"  Fastest operation: {fastest.name} ({fastest.execution_time:.3f}s)")
            print(f"  Slowest operation: {slowest.name} ({slowest.execution_time:.3f}s)")
            print(f"  Performance ratio: {slowest.execution_time/fastest.execution_time:.1f}x")
        
        # Memory usage analysis
        high_memory_ops = [r for r in self.results if r.memory_usage > 100]  # > 100MB
        if high_memory_ops:
            print(f"  High memory operations: {len(high_memory_ops)}")
            avg_memory = np.mean([r.memory_usage for r in high_memory_ops])
            print(f"  Average memory usage: {avg_memory:.1f} MB")
        
        # Parallel efficiency
        parallel_results = [r for r in self.results if 'cores' in r.additional_metrics]
        if len(parallel_results) > 1:
            efficiencies = [r.additional_metrics['efficiency'] for r in parallel_results]
            avg_efficiency = np.mean(efficiencies)
            print(f"  Average parallel efficiency: {avg_efficiency:.1f}%")
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite"""
        print("Starting comprehensive benchmark suite...\n")
        
        start_time = time.time()
        
        self.benchmark_matrix_operations()
        self.benchmark_parallel_scaling()
        self.benchmark_memory_patterns()
        self.benchmark_tensor_operations()
        
        total_time = time.time() - start_time
        
        print(f"Benchmark suite completed in {total_time:.2f} seconds")
        self.generate_report()

def quick_performance_test():
    """Quick performance test for live demos"""
    print("=== Quick Performance Test ===\n")
    
    # Test 1: Matrix multiplication scaling
    print("1. Matrix Multiplication Scaling:")
    sizes = [500, 1000, 1500, 2000]
    times = []
    
    for size in sizes:
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        
        start_time = time.time()
        C = np.dot(A, B)
        exec_time = time.time() - start_time
        times.append(exec_time)
        
        gflops = (2 * size**3) / (exec_time * 1e9)
        print(f"   {size}x{size}: {exec_time:.3f}s ({gflops:.2f} GFLOPS)")
    
    # Test 2: Parallel vs Sequential
    print(f"\n2. Parallel vs Sequential Comparison:")
    
    def cpu_intensive_task(n):
        return np.sum(np.sin(np.arange(n)) * np.cos(np.arange(n)))
    
    n = 5_000_000
    
    # Sequential
    start_time = time.time()
    seq_result = cpu_intensive_task(n)
    seq_time = time.time() - start_time
    
    # Parallel
    def parallel_task(chunk_size):
        return cpu_intensive_task(chunk_size)
    
    num_cores = mp.cpu_count()
    chunk_size = n // num_cores
    
    start_time = time.time()
    with mp.Pool(num_cores) as pool:
        parallel_results = pool.map(parallel_task, [chunk_size] * num_cores)
    parallel_result = sum(parallel_results)
    parallel_time = time.time() - start_time
    
    speedup = seq_time / parallel_time
    efficiency = speedup / num_cores * 100
    
    print(f"   Sequential: {seq_time:.3f}s")
    print(f"   Parallel ({num_cores} cores): {parallel_time:.3f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Efficiency: {efficiency:.1f}%")
    
    # Test 3: Memory bandwidth
    print(f"\n3. Memory Bandwidth Test:")
    
    # Large array operations
    size = 50_000_000  # 50M elements
    data = np.random.rand(size).astype(np.float32)
    
    # Read bandwidth
    start_time = time.time()
    checksum = np.sum(data)
    read_time = time.time() - start_time
    read_bandwidth = (size * 4) / (read_time * 1024**3)  # GB/s
    
    # Write bandwidth
    start_time = time.time()
    data[:] = 1.0
    write_time = time.time() - start_time
    write_bandwidth = (size * 4) / (write_time * 1024**3)  # GB/s
    
    print(f"   Array size: {size/1e6:.1f}M elements ({size*4/1024**2:.1f} MB)")
    print(f"   Read bandwidth: {read_bandwidth:.2f} GB/s")
    print(f"   Write bandwidth: {write_bandwidth:.2f} GB/s")

def stress_test():
    """Stress test for system limits"""
    print(f"\n=== System Stress Test ===\n")
    
    # Memory stress test
    print("1. Memory Stress Test:")
    try:
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024**3)
        
        # Try to allocate 50% of available memory
        target_gb = available_gb * 0.5
        target_elements = int(target_gb * 1024**3 / 8)  # 8 bytes per float64
        
        print(f"   Available memory: {available_gb:.1f} GB")
        print(f"   Attempting to allocate: {target_gb:.1f} GB")
        
        start_time = time.time()
        large_array = np.random.rand(target_elements)
        allocation_time = time.time() - start_time
        
        # Perform operation on large array
        start_time = time.time()
        result = np.sum(large_array**2)
        operation_time = time.time() - start_time
        
        print(f"   Allocation time: {allocation_time:.3f}s")
        print(f"   Operation time: {operation_time:.3f}s")
        print(f"   Memory bandwidth: {target_gb/operation_time:.2f} GB/s")
        
        # Clean up
        del large_array
        
    except MemoryError:
        print("   Memory allocation failed - insufficient memory")
    
    # CPU stress test
    print(f"\n2. CPU Stress Test:")
    num_cores = mp.cpu_count()
    
    def cpu_burn(duration):
        """Burn CPU for specified duration"""
        end_time = time.time() + duration
        operations = 0
        while time.time() < end_time:
            _ = np.sin(np.random.rand(1000))
            operations += 1000
        return operations
    
    duration = 3  # seconds
    print(f"   Running CPU burn test for {duration} seconds on {num_cores} cores...")
    
    start_time = time.time()
    with mp.Pool(num_cores) as pool:
        results = pool.map(cpu_burn, [duration] * num_cores)
    total_time = time.time() - start_time
    
    total_operations = sum(results)
    ops_per_second = total_operations / total_time
    
    print(f"   Total operations: {total_operations:,}")
    print(f"   Operations per second: {ops_per_second:,.0f}")
    print(f"   CPU utilization: {(total_time/duration/num_cores)*100:.1f}%")

def main():
    """Main benchmark function"""
    print("HPC & Tensor Benchmark Suite")
    print("Choose benchmark type:")
    print("1. Quick Performance Test (2-3 minutes)")
    print("2. Full Benchmark Suite (5-10 minutes)")
    print("3. System Stress Test (3-5 minutes)")
    print("4. All Benchmarks")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            quick_performance_test()
        elif choice == '2':
            benchmark = PerformanceBenchmark()
            benchmark.run_full_benchmark()
        elif choice == '3':
            stress_test()
        elif choice == '4':
            quick_performance_test()
            benchmark = PerformanceBenchmark()
            benchmark.run_full_benchmark()
            stress_test()
        else:
            print("Invalid choice, running quick test...")
            quick_performance_test()
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")

if __name__ == "__main__":
    main()