import os
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec






def main():

    category_start = 0
    category_end = category_start + 100

    # Load the residuals from disk and merge them into one tensor
    evaluation_residuals = torch.zeros(100, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters

    for i, start in enumerate(range(category_start, category_end, 10)):
        evaluation_residuals[i:(i+10)] = torch.load(f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {start} sfere_end {start+10}.pt")

    # Compute shared boundary residual as mean
    boundary_residual = torch.mean(evaluation_residuals[:, 3:5], dim=1)



    fig_width = 12.0
    fig_height = 3.0
    left = 0.9
    right = 0.6
    top = 0.4
    bottom = 0.6
    
    fig = plt.figure(figsize=(fig_width, fig_height))


    width_ratios = [6] + [0.1]
    grid_spec = GridSpec(1, 2, width_ratios=width_ratios, figure=fig) # Add one for the color bars

    ax = fig.add_subplot(grid_spec[0, 0])
    
    left   = left   / fig_width
    right  = 1 - right / fig_width
    bottom = bottom / fig_height
    top    = 1 - top / fig_height
    plt.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        hspace=0.15,
        wspace=0.01
    )

    ax.set_title("Average Non-Dimensionalised Residual per Loss Term")
    ax.set_xlabel("Evaluation Iterations")
    ax.set_ylabel("Residual")

    skip_first_entries = 1

    ax.semilogy([i*10 for i in range(skip_first_entries, evaluation_residuals.shape[0])], torch.mean(evaluation_residuals[:,0,skip_first_entries:], dim=0), label="$L_h / S_h$")
    ax.semilogy([i*10 for i in range(skip_first_entries, evaluation_residuals.shape[0])], torch.mean(evaluation_residuals[:,1,skip_first_entries:], dim=0), label="$L_u / S_u$")
    ax.semilogy([i*10 for i in range(skip_first_entries, evaluation_residuals.shape[0])], torch.mean(evaluation_residuals[:,2,skip_first_entries:], dim=0), label="$L_v / S_v$")
    ax.semilogy([i*10 for i in range(skip_first_entries, evaluation_residuals.shape[0])], torch.mean(evaluation_residuals[:,3,skip_first_entries:], dim=0), label="$L_{bound} / S_{bound}$")

    
    plt.legend(loc="upper right")

    os.makedirs(f"./Hybrid Hydro-PINN evaluation/figures", exist_ok=True)
    plt.savefig(f"./Hybrid Hydro-PINN evaluation/figures/Non-dim residual hybrid {category_start}-{category_end}.jpg", dpi=150)


if __name__ == "__main__":
    main()