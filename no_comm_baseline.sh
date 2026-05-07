#!/bin/bash
set -e

LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring" "forced_coordination" "counter_circuit_o_1order")

for layout in "${LAYOUTS[@]}"; do

    echo "TRAINING: $layout"

    python train.py --layout "$layout" --seed 42 --timesteps 200000

    echo ""
    echo "EVALUATING: $layout"

    python evaluate.py --layout "$layout" --seed 42 --episodes 20
done


echo " All 5 layouts done."