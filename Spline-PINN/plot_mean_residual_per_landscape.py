import torch
import os
import matplotlib.pyplot as plt




def main():

    # Load the characteristic scales from disk
    characteristic_scales = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt").unsqueeze(2).repeat(1, 1, 100)
    
    # Load the residuals from disk and merge them into one tensor
    evaluation_loss_terms = torch.zeros(500, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters

    for i in range(0, 500, 10):
        evaluation_loss_terms[i:(i+10)] = torch.load(f"./Saltmarsh component Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt")
        print(torch.load(f"./Saltmarsh component Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt").shape)

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

    plt.figure(figsize=(7, 4))

    # Select which quantity to plot
    quantity = "open_bound"

    if quantity == "h":
        plt.plot(evaluation_residuals[:, 0], label=r"$\mathcal{R}_h$", color="tab:blue")
        
    if quantity == "uv":
        plt.plot(evaluation_residuals[:, 1], label=r"$\mathcal{R}_{uv}$", color="tab:orange")
        
    if quantity == "closed_bound":
        plt.plot(evaluation_residuals[:, 2], label=r"$\mathcal{R}_{bound,closed}$", color="tab:green")
        
    if quantity == "open_bound":
        plt.plot(evaluation_residuals[:, 3], label=r"$\mathcal{R}_{bound,open}$", color="tab:red")
        


    plt.title(r"Residual per Landscape, averaged over 1000 iterations")


    plt.xlabel("Landscape index $n$, sampled at $k(n)$ iterations of SFERE")
    plt.ylabel("Residual")
    plt.legend(loc="upper right", ncols=1)

    os.makedirs(f"./Saltmarsh component Hydro-PINN evaluation/figures", exist_ok=True)
    plt.savefig(f"./Saltmarsh component Hydro-PINN evaluation/figures/Residual per landscape {quantity}.jpg", dpi=150)

    plt.show()



if __name__ == "__main__":
    main()