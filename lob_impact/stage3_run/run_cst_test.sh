#!/bin/bash
# Test CST aggressive scenario: estimate params + run small scenario
set -e

echo "=== Step 1: Estimate CST params from raw LOBSTER data ==="
cd /app

# Check lob_bench is available
ls /app/lob_bench/cst_model/param_estimation.py

python3 -c "
import sys, os
sys.path.insert(0, '/app/lob_bench/cst_model')
sys.path.insert(0, '/app')
os.chdir('/app')
from param_estimation import estimate_from_data_files

params = estimate_from_data_files(
    '/home/myuser/data/rawLOBSTER/GOOG/JAN2023',
    save_path='/home/myuser/scratch/cst_params_goog.pkl',
    tick_size=100,
    num_ticks=500,
)
print('Done! Params saved.')
print('Keys:', list(params.keys()))
"

echo ""
echo "=== Step 2: Run CST aggressive scenario (small test) ==="
cd /app
python3 -u lob_impact/scenarios/4.aggressive_scenario_cst.py \
    --config lob_impact/scenarios/4.aggressive_scenario_cst_config_test.yaml

echo ""
echo "=== Step 3: Verify output ==="
OUTDIR=$(ls -td /home/myuser/scratch/cst_scenario_test/exp_* | head -1)
echo "Output directory: $OUTDIR"
echo ""
echo "--- Files in data_cond/ ---"
ls -la "$OUTDIR/data_cond/" | head -10
echo ""
echo "--- Files in data_gen/ ---"
ls -la "$OUTDIR/data_gen/" | head -10
echo ""
echo "--- aggressive_indices.csv ---"
cat "$OUTDIR/aggressive_indices.csv"
echo ""
echo "--- First 3 lines of a message CSV ---"
head -3 "$OUTDIR/data_gen/"*message*gen* | head -5
echo ""
echo "--- Message CSV column count ---"
head -1 "$OUTDIR/data_gen/"*message*gen* | awk -F',' '{print NF " columns"}'
echo ""
echo "--- Orderbook CSV column count ---"
head -1 "$OUTDIR/data_gen/"*orderbook*gen* | awk -F',' '{print NF " columns"}'
echo ""
echo "=== ALL DONE ==="
