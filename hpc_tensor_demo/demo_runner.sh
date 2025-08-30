#!/bin/bash
# Demo Runner Script
# Convenient script to run demos with proper setup

echo "=============================================="
echo "    HPC & Tensor Operations Demo Runner"
echo "=============================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "8_interactive_demo.py" ]]; then
    echo "Error: Please run this script from the hpc_tensor_demo directory"
    exit 1
fi

# Function to run a specific demo
run_demo() {
    local demo_file=$1
    local demo_name=$2
    
    echo "Running: $demo_name"
    echo "----------------------------------------"
    
    if [[ -f "$demo_file" ]]; then
        python3 "$demo_file"
        echo
        echo "Demo completed. Press Enter to continue..."
        read
    else
        echo "Error: $demo_file not found"
    fi
}

# Main menu
while true; do
    echo "Available demos:"
    echo "  1) Parallel Matrix Multiplication"
    echo "  2) Basic Tensor Operations"
    echo "  3) MPI Distributed Computing"
    echo "  4) OpenMP-style Simulation"
    echo "  5) Advanced Tensor Operations"
    echo "  6) GPU Acceleration"
    echo "  7) Distributed Computing Patterns"
    echo "  8) Interactive Demo (Python menu)"
    echo "  9) Run All Demos"
    echo "  q) Quit"
    echo
    
    read -p "Select demo (1-9, q): " choice
    
    case $choice in
        1)
            run_demo "1_parallel_matrix_multiply.py" "Parallel Matrix Multiplication"
            ;;
        2)
            run_demo "2_tensor_operations.py" "Basic Tensor Operations"
            ;;
        3)
            echo "Running MPI demo with 4 processes..."
            if command -v mpirun &> /dev/null; then
                mpirun -n 4 python3 3_mpi_example.py
                echo "Press Enter to continue..."
                read
            else
                echo "Error: mpirun not found. Please install MPI."
                echo "Ubuntu/Debian: sudo apt-get install openmpi-bin"
                echo "Press Enter to continue..."
                read
            fi
            ;;
        4)
            run_demo "4_openmp_simulation.py" "OpenMP-style Simulation"
            ;;
        5)
            run_demo "5_advanced_tensor_ops.py" "Advanced Tensor Operations"
            ;;
        6)
            run_demo "6_gpu_acceleration.py" "GPU Acceleration"
            ;;
        7)
            run_demo "7_distributed_computing.py" "Distributed Computing Patterns"
            ;;
        8)
            echo "Starting interactive Python demo..."
            python3 8_interactive_demo.py
            ;;
        9)
            echo "Running all demos..."
            for i in {1..7}; do
                if [[ $i -eq 3 ]]; then
                    if command -v mpirun &> /dev/null; then
                        echo "Running MPI demo..."
                        mpirun -n 4 python3 3_mpi_example.py
                    else
                        echo "Skipping MPI demo (mpirun not available)"
                    fi
                else
                    python3 "${i}_"*.py
                fi
                echo "Press Enter for next demo..."
                read
            done
            ;;
        q|Q)
            echo "Thanks for watching the demos!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
    
    echo
done