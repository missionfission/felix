#!/usr/bin/env python3
"""
HPC Demo 4: Distributed Computing Simulation
Demonstrates distributed computing patterns and load balancing
"""

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading

class DistributedTaskManager:
    """Simulates a distributed task management system"""
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.completed_tasks = 0
        self.total_tasks = 0
    
    def add_task(self, task_func, *args):
        """Add a task to the queue"""
        self.task_queue.put((task_func, args))
        self.total_tasks += 1
    
    def worker(self, worker_id):
        """Worker function that processes tasks"""
        while True:
            try:
                task_func, args = self.task_queue.get(timeout=1)
                start_time = time.time()
                result = task_func(*args)
                end_time = time.time()
                
                self.result_queue.put({
                    'worker_id': worker_id,
                    'result': result,
                    'execution_time': end_time - start_time
                })
                
                self.completed_tasks += 1
                self.task_queue.task_done()
                
            except queue.Empty:
                break
    
    def run_distributed(self):
        """Run all tasks in a distributed manner"""
        print(f"Starting {self.num_workers} workers for {self.total_tasks} tasks...")
        
        # Start workers
        threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for completion
        start_time = time.time()
        self.task_queue.join()
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        
        # Wait for threads to finish
        for t in threads:
            t.join()
        
        return results, total_time

def compute_prime_factors(n):
    """CPU-intensive task: find prime factors"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def monte_carlo_integration(n_samples, func_id):
    """Monte Carlo integration of different functions"""
    np.random.seed(func_id * 1000)  # Different seed per task
    
    functions = [
        lambda x: x**2,
        lambda x: np.sin(x),
        lambda x: np.exp(-x**2),
        lambda x: 1/(1 + x**2)
    ]
    
    func = functions[func_id % len(functions)]
    
    # Integration from 0 to 1
    x = np.random.uniform(0, 1, n_samples)
    y = func(x)
    integral = np.mean(y)
    
    return integral, func_id

def distributed_computing_demo():
    """Main distributed computing demonstration"""
    print("=== Distributed Computing Demo ===\n")
    
    # Demo 1: Prime factorization
    print("1. Distributed Prime Factorization:")
    manager = DistributedTaskManager(num_workers=4)
    
    # Add prime factorization tasks
    large_numbers = [982451653, 982451654, 982451655, 982451656, 
                    982451657, 982451658, 982451659, 982451660]
    
    for num in large_numbers:
        manager.add_task(compute_prime_factors, num)
    
    results, total_time = manager.run_distributed()
    
    print(f"   Completed {len(results)} factorizations in {total_time:.3f} seconds")
    for i, result in enumerate(results[:3]):  # Show first 3 results
        worker_id = result['worker_id']
        factors = result['result']
        exec_time = result['execution_time']
        number = large_numbers[i]
        print(f"   Worker {worker_id}: {number} = {' × '.join(map(str, factors))} ({exec_time:.3f}s)")
    
    # Demo 2: Monte Carlo integration
    print(f"\n2. Distributed Monte Carlo Integration:")
    manager2 = DistributedTaskManager(num_workers=6)
    
    # Add integration tasks
    n_samples = 1_000_000
    for i in range(12):  # 12 different integration tasks
        manager2.add_task(monte_carlo_integration, n_samples, i)
    
    results2, total_time2 = manager2.run_distributed()
    
    print(f"   Completed {len(results2)} integrations in {total_time2:.3f} seconds")
    print(f"   Average execution time per task: {total_time2/len(results2):.3f} seconds")
    
    # Group results by function
    function_names = ["x²", "sin(x)", "exp(-x²)", "1/(1+x²)"]
    for func_id in range(4):
        func_results = [r for r in results2 if r['result'][1] % 4 == func_id]
        if func_results:
            integrals = [r['result'][0] for r in func_results]
            mean_integral = np.mean(integrals)
            std_integral = np.std(integrals)
            print(f"   {function_names[func_id]}: {mean_integral:.6f} ± {std_integral:.6f}")

def map_reduce_simulation():
    """Simulate MapReduce pattern for big data processing"""
    print(f"\n=== MapReduce Simulation ===\n")
    
    def map_function(data_chunk):
        """Map phase: process a chunk of data"""
        word_counts = {}
        for text in data_chunk:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        return word_counts
    
    def reduce_function(mapped_results):
        """Reduce phase: combine results from all mappers"""
        total_counts = {}
        for word_counts in mapped_results:
            for word, count in word_counts.items():
                total_counts[word] = total_counts.get(word, 0) + count
        return total_counts
    
    # Generate sample text data
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning and artificial intelligence",
        "high performance computing with parallel processing",
        "tensor operations and matrix multiplication",
        "distributed systems and cloud computing",
        "data science and big data analytics"
    ] * 1000  # Repeat to create larger dataset
    
    print(f"Processing {len(sample_texts)} text documents")
    
    # Split data into chunks for parallel processing
    num_workers = 4
    chunk_size = len(sample_texts) // num_workers
    data_chunks = [
        sample_texts[i*chunk_size:(i+1)*chunk_size] 
        for i in range(num_workers)
    ]
    
    # Map phase (parallel)
    print(f"\n1. Map phase ({num_workers} workers):")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        map_results = list(executor.map(map_function, data_chunks))
    
    map_time = time.time() - start_time
    print(f"   Map phase completed in {map_time:.3f} seconds")
    
    # Reduce phase
    print(f"\n2. Reduce phase:")
    start_time = time.time()
    final_counts = reduce_function(map_results)
    reduce_time = time.time() - start_time
    
    print(f"   Reduce phase completed in {reduce_time:.4f} seconds")
    print(f"   Total unique words: {len(final_counts)}")
    
    # Show top words
    sorted_words = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"   Top 10 words:")
    for word, count in sorted_words[:10]:
        print(f"     '{word}': {count}")
    
    print(f"\n3. Total MapReduce time: {map_time + reduce_time:.3f} seconds")

def load_balancing_demo():
    """Demonstrate dynamic load balancing"""
    print(f"\n=== Dynamic Load Balancing Demo ===\n")
    
    def variable_workload_task(task_id):
        """Task with variable computational complexity"""
        # Simulate different workloads
        complexity = np.random.randint(1, 6)  # 1-5 complexity levels
        n_operations = complexity * 100000
        
        start_time = time.time()
        # Simulate work with different computational intensity
        result = 0
        for i in range(n_operations):
            result += np.sin(i) * np.cos(i)
        
        execution_time = time.time() - start_time
        return {
            'task_id': task_id,
            'complexity': complexity,
            'result': result,
            'execution_time': execution_time
        }
    
    # Create tasks with variable complexity
    num_tasks = 20
    print(f"Creating {num_tasks} tasks with variable complexity...")
    
    # Static load balancing (equal task distribution)
    print(f"\n1. Static load balancing:")
    start_time = time.time()
    
    num_workers = 4
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        static_results = list(executor.map(variable_workload_task, range(num_tasks)))
    
    static_time = time.time() - start_time
    
    # Analyze worker utilization
    worker_times = [0] * num_workers
    for i, result in enumerate(static_results):
        worker_id = i % num_workers
        worker_times[worker_id] += result['execution_time']
    
    print(f"   Total time: {static_time:.3f} seconds")
    print(f"   Worker utilization:")
    for i, wt in enumerate(worker_times):
        print(f"     Worker {i}: {wt:.3f}s ({wt/max(worker_times)*100:.1f}%)")
    
    # Dynamic load balancing (work stealing simulation)
    print(f"\n2. Dynamic load balancing simulation:")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and process as they complete
        futures = {executor.submit(variable_workload_task, i): i for i in range(num_tasks)}
        dynamic_results = []
        
        for future in as_completed(futures):
            result = future.result()
            dynamic_results.append(result)
    
    dynamic_time = time.time() - start_time
    
    print(f"   Total time: {dynamic_time:.3f} seconds")
    print(f"   Improvement: {(static_time - dynamic_time)/static_time*100:.1f}%")
    
    # Show complexity distribution
    complexities = [r['complexity'] for r in dynamic_results]
    print(f"   Task complexity distribution: {np.bincount(complexities)[1:]}")

def main():
    """Main demo function"""
    distributed_computing_demo()
    map_reduce_simulation()
    load_balancing_demo()

if __name__ == "__main__":
    main()