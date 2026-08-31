import torch
import os
import matplotlib.pyplot as plt




def main():

    # Load the characteristic scales from disk
    characteristic_scales = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt").unsqueeze(2).repeat(1, 1, 100)
    
    # Load the residuals from disk and merge them into one tensor
    evaluation_loss_terms = torch.zeros(500, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters

    for i in range(0, 500, 10):
        evaluation_loss_terms[i:(i+10)] = torch.load(f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt")
        print(torch.load(f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt").shape)

    # Compute residuals
    evaluation_residuals = torch.zeros(500, 4, 100)

    # Index 0 - h residual
    evaluation_residuals[:, 0, :] = evaluation_loss_terms[:, 0, :] / characteristic_scales[:, 0, :]

    # Index 1 - uv residual
    evaluation_residuals[:, 1, :] = (evaluation_loss_terms[:, 1 ,:] + evaluation_loss_terms[:, 2, :]) / (characteristic_scales[:, 1, :] + characteristic_scales[:, 2, :])

    # Index 2 - closed boundary residual
    evaluation_residuals[:, 2, :] = evaluation_loss_terms[:, 3, :] / characteristic_scales[:, 3, :]
    
    # Index 3 - closed boundary residual
    evaluation_residuals[:, 3, :] = evaluation_loss_terms[:, 4, :] / characteristic_scales[:, 4, :]

    # Compute aggregate evaluation residual per landscape
    evaluation_residuals = torch.mean(evaluation_residuals, dim=2)

    # Load the characteristic scales from disk
    characteristic_scales = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt")

    plt.figure(figsize=(7, 4))

    # Select which quantity to plot
    quantity = "h"

    if quantity == "h":
        plt.scatter(characteristic_scales[:, 0], evaluation_residuals[:, 0], label=r"$\mathcal{R}_h$", color="tab:blue", s=10)
        
    if quantity == "uv":
        plt.scatter((characteristic_scales[:, 1] + characteristic_scales[:, 2]), evaluation_residuals[:, 1], label=r"$\mathcal{R}_{uv}$", color="tab:orange", s=10)
        
    if quantity == "closed_bound":
        plt.scatter(characteristic_scales[:, 3], evaluation_residuals[:, 2], label=r"$\mathcal{R}_{bound,closed}$", color="tab:green", s=10)
        
    if quantity == "open_bound":
        plt.scatter(characteristic_scales[:, 4], evaluation_residuals[:, 3], label=r"$\mathcal{R}_{bound,open}$", color="tab:red", s=10)
        


    plt.title(r"Residual against Characteristic Scale per Landscape")


    plt.xlabel("Landscape Characteristic Scale")
    plt.ylabel("Residual")
    plt.legend(loc="upper right", ncols=1)

    os.makedirs(f"./Hybrid Hydro-PINN evaluation/figures", exist_ok=True)
    plt.savefig(f"./Hybrid Hydro-PINN evaluation/figures/Scatter plot residual against c-scale {quantity}.jpg", dpi=150)

    plt.show()



if __name__ == "__main__":
    main()