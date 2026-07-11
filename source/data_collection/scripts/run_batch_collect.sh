#!/bin/bash
# Batch data collection: loop geniesim autocollect run, restart container each batch
# Usage: bash run_batch_collect.sh <TASK> <REPEATS>
# Example: bash run_batch_collect.sh lab_dual_arm_sample_transfer_g2 100

set +eo pipefail

TASK="${1:?Usage: $0 <TASK> <REPEATS>}"
REPEATS="${2:?Usage: $0 <TASK> <REPEATS>}"

echo "=========================================="
echo "Batch Data Collection"
echo "  Task:    $TASK"
echo "  Repeats: $REPEATS"
echo "=========================================="

for i in $(seq 1 "$REPEATS"); do
    echo ""
    echo "--- Batch $i / $REPEATS ---"

    geniesim autocollect run "$TASK" --headless --standalone

    docker stop data_collection_open_source 2>/dev/null || true
    docker rm data_collection_open_source 2>/dev/null || true

    DC_ROOT="/home/zhangmingfa/embodiedai/genie_sim/source/data_collection"
    COUNT=$(ls -d "$DC_ROOT/recording_data/"* 2>/dev/null | wc -l)
    echo "  Recordings so far: $COUNT"

    sleep 5
done

echo ""
echo "Done! Ran $REPEATS batches."
