#!/bin/bash
#
# Run All Benchmarks for Kitsune Optimizer
# This script runs MLP, LeNet-5, and ResNet-18 benchmarks
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Kitsune Benchmark Suite"
echo "========================================"
echo ""

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA GPU required."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Check Python and packages
echo "Checking dependencies..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import kitsune; print(f'Kitsune: {kitsune.__version__}')"
echo ""

# Parse arguments
RUNS=5
ITERATIONS=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --runs N          Number of benchmark runs (default: 5)"
            echo "  --iterations N    Iterations per run (default: 100)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Runs: $RUNS"
echo "  Iterations: $ITERATIONS"
echo ""

# Run benchmarks
echo "========================================"
echo "1/3: Running MLP Benchmark"
echo "========================================"
python3 benchmark_mlp.py --runs "$RUNS" --iterations "$ITERATIONS"
echo ""

echo "========================================"
echo "2/3: Running LeNet-5 Benchmark"
echo "========================================"
python3 benchmark_lenet5.py --runs "$RUNS" --iterations "$ITERATIONS"
echo ""

echo "========================================"
echo "3/3: Running ResNet-18 Benchmark"
echo "========================================"
python3 benchmark_resnet18.py --runs "$RUNS" --iterations "$ITERATIONS"
echo ""

# Generate visualizations
echo "========================================"
echo "Generating Visualizations"
echo "========================================"
python3 visualize_results.py
echo ""

echo "========================================"
echo "Benchmark Suite Complete!"
echo "========================================"
echo ""
echo "Results saved to: ../results/"
echo "Charts saved to: ../charts/"
echo ""
echo "View summary in: ../README.md"
