#!/usr/bin/env zsh


python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='0-100' --sfere_start=0 --sfere_end=50
python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='0-100' --sfere_start=50 --sfere_end=100

python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='100-200' --sfere_start=100 --sfere_end=150
python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='100-200' --sfere_start=150 --sfere_end=200

python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='200-300' --sfere_start=200 --sfere_end=250
python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='200-300' --sfere_start=250 --sfere_end=300

python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='300-400' --sfere_start=300 --sfere_end=350
python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='300-400' --sfere_start=350 --sfere_end=400

python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='400-500' --sfere_start=400 --sfere_end=450
python main_evaluate_residuals.py --resolution_factor=4 --ablation_model='400-500' --sfere_start=450 --sfere_end=500