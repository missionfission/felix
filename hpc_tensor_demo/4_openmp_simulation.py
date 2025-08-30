#!/usr/bin/env python3
"""
HPC Demo 3: OpenMP-style Parallel Computing Simulation
Demonstrates thread-based parallelism for HPC workloads
"""

import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

class ParallelComputation:
    """Class to demonstrate parallel computation patterns"""
    
    def __init__(self, num_threads=None):
        self.num_threads = num_threads or mp.cpu_count()
        self.results = {}
        self.lock = threading.Lock()
    
    def compute_pi_monte_carlo(self, n_samples):
        """Compute π using Monte Carlo method - embarrassingly parallel"""
        def worker(thread_id, samples_per_thread):
            np.random.seed(thread_id)  # Different seed per thread
            x = np.random.uniform(-1, 1, samples_per_thread)
            y = np.random.uniform(-1, 1, samples_per_thread)
            inside_circle = np.sum(x**2 + y**2 <= 1)
            
            with self.lock:
                self.results[thread_id] = inside_circle
        
        print(f"=== Computing π using Monte Carlo ({n_samples:,} samples) ===")
        print(f"Using {self.num_threads} threads\n")
        
        samples_per_thread = n_samples // self.num_threads
        
        # Parallel execution
        start_time = time.time()
        threads = []
        
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=worker, 
                args=(i, samples_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        parallel_time = time.time() - start_time
        
        # Calculate π
        total_inside = sum(self.results.values())
        pi_estimate = 4 * total_inside / n_samples
        
        print(f"Parallel computation:")
        print(f"   Time: {parallel_time:.3f} seconds")
        print(f"   π estimate: {pi_estimate:.6f}")
        print(f"   Error: {abs(pi_estimate - np.pi):.6f}")
        
        # Sequential comparison
        self.results.clear()
        start_time = time.time()
        np.random.seed(42)
        x = np.random.uniform(-1, 1, n_samples)
        y = np.random.uniform(-1, 1, n_samples)
        inside_circle_seq = np.sum(x**2 + y**2 <= 1)
        sequential_time = time.time() - start_time
        pi_seq = 4 * inside_circle_seq / n_samples
        
        print(f"\nSequential computation:")
        print(f"   Time: {sequential_time:.3f} seconds")
        print(f"   π estimate: {pi_seq:.6f}")
        print(f"   Speedup: {sequential_time/parallel_time:.2f}x")

def parallel_numerical_integration():
    """Parallel numerical integration using thread pool"""
    print(f"\n=== Parallel Numerical Integration ===\n")
    
    def integrand(x):
        """Function to integrate: sin(x) * exp(-x)"""
        return np.sin(x) * np.exp(-x)
    
    def integrate_chunk(start, end, n_points):
        """Integrate a chunk of the domain"""
        x = np.linspace(start, end, n_points)
        y = integrand(x)
        # Trapezoidal rule
        return np.trapz(y, x)
    
    # Integration parameters
    a, b = 0, 10  # Integration bounds
    n_total = 10_000_000
    num_workers = mp.cpu_count()
    
    print(f"Integrating sin(x)*exp(-x) from {a} to {b}")
    print(f"Total points: {n_total:,}, Workers: {num_workers}")
    
    # Parallel integration
    start_time = time.time()
    chunk_size = (b - a) / num_workers
    n_per_chunk = n_total // num_workers
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            start = a + i * chunk_size
            end = start + chunk_size
            future = executor.submit(integrate_chunk, start, end, n_per_chunk)
            futures.append(future)
        
        parallel_result = sum(future.result() for future in as_completed(futures))
    
    parallel_time = time.time() - start_time
    
    # Sequential integration
    start_time = time.time()
    x = np.linspace(a, b, n_total)
    y = integrand(x)
    sequential_result = np.trapz(y, x)
    sequential_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"   Parallel result: {parallel_result:.8f}")
    print(f"   Sequential result: {sequential_result:.8f}")
    print(f"   Difference: {abs(parallel_result - sequential_result):.2e}")
    print(f"   Parallel time: {parallel_time:.3f} seconds")
    print(f"   Sequential time: {sequential_time:.3f} seconds")
    print(f"   Speedup: {sequential_time/parallel_time:.2f}x")

def simulate_particle_system():
    """Simulate a particle system with parallel force calculations"""
    print(f"\n=== Parallel Particle Simulation ===\n")
    
    class ParticleSystem:
        def __init__(self, n_particles):
            self.n_particles = n_particles
            self.positions = np.random.rand(n_particles, 2) * 10
            self.velocities = np.random.rand(n_particles, 2) * 0.1
            self.masses = np.random.rand(n_particles) * 5 + 1
        
        def compute_forces_parallel(self):
            """Compute gravitational forces between particles in parallel"""
            forces = np.zeros_like(self.positions)
            
            def compute_force_chunk(start_idx, end_idx):
                chunk_forces = np.zeros((end_idx - start_idx, 2))
                for i in range(start_idx, end_idx):
                    for j in range(self.n_particles):
                        if i != j:
                            r_vec = self.positions[j] - self.positions[i]
                            r_mag = np.linalg.norm(r_vec)
                            if r_mag > 0.1:  # Avoid singularity
                                F = self.masses[i] * self.masses[j] / (r_mag**3)
                                chunk_forces[i - start_idx] += F * r_vec
                return start_idx, end_idx, chunk_forces
            
            # Parallel force computation
            num_workers = min(4, self.n_particles)
            chunk_size = self.n_particles // num_workers
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i in range(num_workers):
                    start = i * chunk_size
                    end = start + chunk_size if i < num_workers - 1 else self.n_particles
                    future = executor.submit(compute_force_chunk, start, end)
                    futures.append(future)
                
                for future in as_completed(futures):
                    start_idx, end_idx, chunk_forces = future.result()
                    forces[start_idx:end_idx] = chunk_forces
            
            return forces
        
        def update(self, dt=0.01):
            """Update particle positions and velocities"""
            forces = self.compute_forces_parallel()
            
            # Update velocities and positions
            accelerations = forces / self.masses.reshape(-1, 1)
            self.velocities += accelerations * dt
            self.positions += self.velocities * dt
        
        def get_kinetic_energy(self):
            """Calculate total kinetic energy"""
            return 0.5 * np.sum(self.masses.reshape(-1, 1) * self.velocities**2)
    
    # Run simulation
    n_particles = 100
    n_steps = 50
    
    print(f"Simulating {n_particles} particles for {n_steps} steps")
    
    system = ParticleSystem(n_particles)
    
    start_time = time.time()
    initial_energy = system.get_kinetic_energy()
    
    for step in range(n_steps):
        system.update()
        if step % 10 == 0:
            energy = system.get_kinetic_energy()
            print(f"   Step {step:2d}: KE = {energy:.3f}")
    
    simulation_time = time.time() - start_time
    final_energy = system.get_kinetic_energy()
    
    print(f"\nSimulation completed in {simulation_time:.3f} seconds")
    print(f"Initial kinetic energy: {initial_energy:.3f}")
    print(f"Final kinetic energy: {final_energy:.3f}")
    print(f"Energy change: {((final_energy - initial_energy)/initial_energy)*100:.2f}%")

def main():
    """Main demo function"""
    # Monte Carlo π computation
    computer = ParallelComputation()
    computer.compute_pi_monte_carlo(10_000_000)
    
    # Numerical integration
    parallel_numerical_integration()
    
    # Particle simulation
    simulate_particle_system()

if __name__ == "__main__":
    main()