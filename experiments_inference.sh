#!/bin/bash
# filepath: run_generation_tests.sh

# Array of (batch_size, n_samples) pairs
configs=(
    # "16 256"
    # "32 512"
    # "64 512"
    # "256 2048"
    # "512 2048"
    # "1024 2048"
    # "2048 4096"
    "4096 8192"
)

# Run each configuration
for config in "${configs[@]}"; do
    read batch_size n_samples <<< "$config"
    
    echo "=========================================="
    echo "Running with batch_size=$batch_size, n_samples=$n_samples"
    echo "=========================================="
    

    python3 run_inference.py \
        --stock GOOG \
        --test_split 0\
        --checkpoint_step 23 \
        --batch_size $batch_size \
        --n_sequences $n_samples


    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: batch_size=$batch_size, n_samples=$n_samples"
    else
        echo "✗ Failed: batch_size=$batch_size, n_samples=$n_samples"
    fi
    echo ""
done

echo "=========================================="
echo "All tests completed!"
echo "=========================================="