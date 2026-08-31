import torch
import os
import matplotlib.pyplot as plt




def main():

    # Load the characteristic scales from disk
    characteristic_scales = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt")
    
    # Load the residuals from disk and merge them into one tensor
    evaluation_residuals = torch.zeros(500, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters

    for i in range(0, 500, 10):
        evaluation_residuals[i:(i+10)] = torch.load(f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt")
        print(torch.load(f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt").shape)

    # Compute aggregate evaluation residual per landscape
    evaluation_residuals = torch.mean(evaluation_residuals, dim=2)

    # Merge the two momentum channels (direction distinction has been removed by evaluating across all landscape orientations)
    vel_residuals = torch.mean(evaluation_residuals[:, [1,2]], dim=1)

    plt.figure(figsize=(7, 4))

    # Select which quantity to plot
    quantity = "open_bound"

    if quantity == "h":
        plt.plot(evaluation_residuals[:, 0], label=r"$\mathcal{R}_h$", color="tab:blue")
        
    if quantity == "uv":
        plt.plot(vel_residuals[:], label=r"$\mathcal{R}_{uv}$", color="tab:orange")
        
    if quantity == "closed_bound":
        plt.plot(evaluation_residuals[:, 3], label=r"$\mathcal{R}_{bound,closed}$", color="tab:green")
        
    if quantity == "open_bound":
        plt.plot(evaluation_residuals[:, 4], label=r"$\mathcal{R}_{bound,open}$", color="tab:red")
        


    plt.title(r"Residual per Landscape, averaged over 1000 iterations")


    plt.xlabel("Landscape index $n$, sampled at $k(n)$ iterations of SFERE")
    plt.ylabel("Residual")
    plt.legend(loc="upper right", ncols=1)

    os.makedirs(f"./Hybrid Hydro-PINN evaluation/figures", exist_ok=True)
    plt.savefig(f"./Hybrid Hydro-PINN evaluation/figures/Residual per landscape {quantity}.jpg", dpi=150)

    plt.show()



if __name__ == "__main__":
    main()