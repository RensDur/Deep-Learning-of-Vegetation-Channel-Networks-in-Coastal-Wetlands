import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec




def main():

    category_start = 0
    category_end = category_start + 100

    # Load the residuals from disk and merge them into one tensor
    evaluation_residuals = torch.zeros(100, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters

    for i, start in enumerate(range(category_start, category_end, 10)):
        evaluation_residuals[i:(i+10)] = torch.load(f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {start} sfere_end {start+10}.pt")

    # Merge the two momentum channels (direction distinction has been removed by evaluating across all landscape orientations)
    vel_residuals = torch.mean(evaluation_residuals[:, [1,2]], dim=1)

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

    ax.semilogy([i*10 for i in range(skip_first_entries, evaluation_residuals.shape[0])], torch.mean(evaluation_residuals[:,0,skip_first_entries:], dim=0), label=r"$\mathcal{R}_h$")
    ax.semilogy([i*10 for i in range(skip_first_entries, evaluation_residuals.shape[0])], torch.mean(vel_residuals[:,skip_first_entries:], dim=0), label=r"$\mathcal{R}_{uv}$")
    ax.semilogy([i*10 for i in range(skip_first_entries, evaluation_residuals.shape[0])], torch.mean(evaluation_residuals[:,3,skip_first_entries:], dim=0), label=r"$\mathcal{R}_{bound,closed}$")
    ax.semilogy([i*10 for i in range(skip_first_entries, evaluation_residuals.shape[0])], torch.mean(evaluation_residuals[:,4,skip_first_entries:], dim=0), label=r"$\mathcal{R}_{bound,open}$")

    
    plt.legend(loc="upper right")

    os.makedirs(f"./Hybrid Hydro-PINN evaluation/figures", exist_ok=True)
    plt.savefig(f"./Hybrid Hydro-PINN evaluation/figures/Non-dim residual hybrid {category_start}-{category_end}.jpg", dpi=150)

    # plt.show()




def compute_mean_residual_for_model(category_start):

    category_end = category_start + 100

    # Load the residuals from disk and merge them into one tensor
    evaluation_residuals = torch.zeros(100, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters

    for i, start in enumerate(range(category_start, category_end, 10)):
        evaluation_residuals[i:(i+10)] = torch.load(f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {start} sfere_end {start+10}.pt")

    # Merge the two momentum channels (direction distinction has been removed by evaluating across all landscape orientations)
    vel_residuals = torch.mean(evaluation_residuals[:, [1,2]], dim=1)

    # Compute mean per channel
    mean_h_residual = torch.mean(evaluation_residuals[:, 0, 1:])
    mean_vel_residual = torch.mean(vel_residuals[:, 1:])
    mean_closed_bound_residual = torch.mean(evaluation_residuals[:, 3, 1:])
    mean_open_bound_residual = torch.mean(evaluation_residuals[:, 4, 1:])

    return mean_h_residual, mean_vel_residual, mean_closed_bound_residual, mean_open_bound_residual


def main_boxplot(quantity):

    h_residuals = []
    vel_residuals = []
    closed_bound_residuals = []
    open_bound_residuals = []

    for i in range(0, 500, 100):
        print(f"Residuals for model {i}-{i+100}")

        h,vel,closed_bound,open_bound = compute_mean_residual_for_model(i)

        h_residuals.append(h)
        vel_residuals.append(vel)
        closed_bound_residuals.append(closed_bound)
        open_bound_residuals.append(open_bound)

    xs = np.array(["0-100", "100-200", "200-300", "300-400", "400-500"])
    h_residuals = np.array(h_residuals)
    vel_residuals = np.array(vel_residuals)
    closed_bound_residuals = np.array(closed_bound_residuals)
    open_bound_residuals = np.array(open_bound_residuals)
    

    ablation_models = np.array(["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"])

    # residuals_per_category = {
    #     # r"$\mathcal{R}_h$": h_residuals,
    #     # r"$\mathcal{R}_{uv}$": vel_residuals,
    #     r"$\mathcal{R}_{bound,closed}$": closed_bound_residuals,
    #     # r"$\mathcal{R}_{bound,open}$": open_bound_residuals
    # }

    # Extract matplotlib default categorical colours
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(7, 4))

    if quantity == "h":
        plt.bar(ablation_models, h_residuals, label=r"$\mathcal{R}_h$", color=colors[0])
        plt.title(r"Water Level Residual ($\mathcal{R}_h$) per Landscape Category")

    elif quantity == "uv":
        plt.bar(ablation_models, vel_residuals, label=r"$\mathcal{R}_{uv}$", color=colors[1])
        plt.title(r"Momentum Residual ($\mathcal{R}_{uv}$) per Landscape Category")

    elif quantity == "closed_bound":
        plt.bar(ablation_models, closed_bound_residuals, label=r"$\mathcal{R}_{bound,closed}$", color=colors[2])
        plt.title(r"Closed Boundary Residual ($\mathcal{R}_{bound,closed}$) per Landscape Category")

    elif quantity == "open_bound":
        plt.bar(ablation_models, open_bound_residuals, label=r"$\mathcal{R}_{bound,open}$", color=colors[3])
        plt.title(r"Open Boundary Residual ($\mathcal{R}_{bound,open}$) per Landscape Category")

    
    plt.xlabel("Landscape Category")
    plt.ylabel("Non-Dimensionalised Residual")
    plt.legend(loc="upper right", ncols=1)

    os.makedirs(f"./Hybrid Hydro-PINN evaluation/figures", exist_ok=True)
    plt.savefig(f"./Hybrid Hydro-PINN evaluation/figures/Non-dim residual bar per landscape category {quantity}.jpg", dpi=150)

    plt.show()


if __name__ == "__main__":

    # main()
    main_boxplot("h")