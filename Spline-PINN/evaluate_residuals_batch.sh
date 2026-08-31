#!/usr/bin/env zsh




for ((start=0; start<500; start+=10)); do
    end=$((start + 10))
    python main_evaluate_residuals.py --resolution_factor=4 --sfere_start=$start --sfere_end=$end
done