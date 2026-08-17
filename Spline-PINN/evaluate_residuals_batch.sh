#!/usr/bin/env zsh




for ((start=0; start<100; start+=10)); do
    end=$((start + 10))
    python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='0-100' --sfere_start=$start --sfere_end=$end
done

for ((start=100; start<200; start+=10)); do
    end=$((start + 10))
    python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='100-200' --sfere_start=$start --sfere_end=$end
done

for ((start=200; start<300; start+=10)); do
    end=$((start + 10))
    python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='200-300' --sfere_start=$start --sfere_end=$end
done

for ((start=300; start<400; start+=10)); do
    end=$((start + 10))
    python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='300-400' --sfere_start=$start --sfere_end=$end
done

for ((start=400; start<500; start+=10)); do
    end=$((start + 10))
    python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='400-500' --sfere_start=$start --sfere_end=$end
done