#!/usr/bin/env python3
"""
Interactive Demo Script
Main script to run all demos with user interaction
"""

import sys
import time

def print_banner():
    """Print demo banner"""
    print("=" * 60)
    print("    HPC & TENSOR OPERATIONS LIVE DEMO")
    print("=" * 60)
    print()

def print_menu():
    """Print demo menu"""
    demos = [
        "1. Parallel Matrix Multiplication (multiprocessing)",
        "2. Basic Tensor Operations (NumPy/PyTorch)",
        "3. MPI Distributed Computing",
        "4. OpenMP-style Parallel Simulation",
        "5. Advanced Tensor Operations",
        "6. GPU Acceleration (CuPy/PyTorch)",
        "7. Distributed Computing Patterns",
        "8. Run All Demos",
        "9. Exit"
    ]
    
    print("Available Demos:")
    for demo in demos:
        print(f"  {demo}")
    print()

def run_demo(choice):
    """Run the selected demo"""
    demos = {
        '1': ('1_parallel_matrix_multiply.py', 'Parallel Matrix Multiplication'),
        '2': ('2_tensor_operations.py', 'Basic Tensor Operations'),
        '3': ('3_mpi_example.py', 'MPI Distributed Computing'),
        '4': ('4_openmp_simulation.py', 'OpenMP-style Simulation'),
        '5': ('5_advanced_tensor_ops.py', 'Advanced Tensor Operations'),
        '6': ('6_gpu_acceleration.py', 'GPU Acceleration'),
        '7': ('7_distributed_computing.py', 'Distributed Computing'),
    }
    
    if choice in demos:
        script, name = demos[choice]
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        
        try:
            # Import and run the demo
            module_name = script[:-3]  # Remove .py extension
            module = __import__(module_name)
            
            start_time = time.time()
            module.main()
            end_time = time.time()
            
            print(f"\n{'='*50}")
            print(f"Demo completed in {end_time - start_time:.3f} seconds")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"Error running demo: {e}")
            print("Make sure all dependencies are installed (see requirements.txt)")
    
    elif choice == '8':
        print("\n" + "="*50)
        print("RUNNING ALL DEMOS")
        print("="*50)
        
        for demo_choice in ['1', '2', '3', '4', '5', '6', '7']:
            run_demo(demo_choice)
            print("\nPress Enter to continue to next demo...")
            input()
    
    elif choice == '9':
        print("Thanks for watching the demo!")
        return False
    
    else:
        print("Invalid choice. Please try again.")
    
    return True

def check_dependencies():
    """Check which optional dependencies are available"""
    print("Checking dependencies...")
    
    dependencies = {
        'numpy': 'numpy',
        'torch': 'torch',
        'mpi4py': 'mpi4py',
        'cupy': 'cupy'
    }
    
    available = {}
    for name, module in dependencies.items():
        try:
            __import__(module)
            available[name] = True
            print(f"  ✓ {name}")
        except ImportError:
            available[name] = False
            print(f"  ✗ {name} (optional)")
    
    print()
    
    if not available['numpy']:
        print("WARNING: NumPy is required for most demos!")
        print("Install with: pip install numpy")
    
    if not available['torch']:
        print("Note: Some tensor demos require PyTorch")
        print("Install with: pip install torch")
    
    if not available['mpi4py']:
        print("Note: MPI demo requires mpi4py and MPI implementation")
        print("Install with: pip install mpi4py")
    
    print()

def main():
    """Main interactive demo function"""
    print_banner()
    check_dependencies()
    
    while True:
        print_menu()
        choice = input("Select a demo (1-9): ").strip()
        
        if not run_demo(choice):
            break
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()