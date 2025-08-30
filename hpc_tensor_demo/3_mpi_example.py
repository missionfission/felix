#!/usr/bin/env python3
"""
HPC Demo 2: MPI (Message Passing Interface) Example
Demonstrates distributed computing across multiple processes
Run with: mpirun -n 4 python 3_mpi_example.py
"""

try:
    from mpi4py import MPI
    import numpy as np
    import time
    
    def mpi_matrix_multiply():
        """Distributed matrix multiplication using MPI"""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if rank == 0:
            print(f"=== MPI Demo: Distributed Matrix Multiplication ===")
            print(f"Running on {size} processes\n")
            
            # Create matrices on root process
            N = 1000
            A = np.random.rand(N, N)
            B = np.random.rand(N, N)
            print(f"Created {N}x{N} matrices")
            
            # Calculate chunk size
            chunk_size = N // size
            start_time = time.time()
        else:
            A = None
            B = None
            chunk_size = None
        
        # Broadcast B to all processes
        B = comm.bcast(B, root=0)
        chunk_size = comm.bcast(chunk_size, root=0)
        
        # Scatter A rows to different processes
        if rank == 0:
            A_chunks = [A[i*chunk_size:(i+1)*chunk_size] for i in range(size)]
            # Handle remainder
            if N % size != 0:
                A_chunks[-1] = A[(size-1)*chunk_size:]
        else:
            A_chunks = None
        
        A_local = comm.scatter(A_chunks, root=0)
        
        # Perform local computation
        C_local = np.dot(A_local, B)
        
        # Gather results
        C_chunks = comm.gather(C_local, root=0)
        
        if rank == 0:
            # Combine results
            C = np.vstack(C_chunks)
            end_time = time.time()
            
            print(f"MPI computation completed in {end_time - start_time:.3f} seconds")
            print(f"Result matrix shape: {C.shape}")
            print(f"Sample result (top-left 3x3):\n{C[:3, :3]}")
            
            # Compare with sequential
            print("\nComparing with sequential computation...")
            start_time = time.time()
            C_seq = np.dot(A, B)
            seq_time = time.time() - start_time
            print(f"Sequential time: {seq_time:.3f} seconds")
            
            if np.allclose(C, C_seq):
                print("✓ Results match!")
            else:
                print("✗ Results don't match!")
    
    def mpi_reduce_example():
        """Example of MPI reduce operations"""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if rank == 0:
            print(f"\n=== MPI Reduce Example ===")
        
        # Each process generates some data
        local_data = np.random.rand(1000) * (rank + 1)
        local_sum = np.sum(local_data)
        
        print(f"Process {rank}: local sum = {local_sum:.3f}")
        
        # Reduce to get global sum
        global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
        
        if rank == 0:
            print(f"Global sum across all processes: {global_sum:.3f}")
    
    def main():
        """Main MPI demo function"""
        mpi_matrix_multiply()
        mpi_reduce_example()
    
    if __name__ == "__main__":
        main()

except ImportError:
    print("MPI4Py not available. Install with: pip install mpi4py")
    print("Also requires MPI implementation (e.g., OpenMPI, MPICH)")
    print("\nTo run this demo:")
    print("1. Install MPI: sudo apt-get install openmpi-bin openmpi-dev")
    print("2. Install mpi4py: pip install mpi4py")
    print("3. Run with: mpirun -n 4 python 3_mpi_example.py")