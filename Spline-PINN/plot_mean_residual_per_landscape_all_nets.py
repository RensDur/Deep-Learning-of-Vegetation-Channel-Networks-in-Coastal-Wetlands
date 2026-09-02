import torch
import os
import matplotlib.pyplot as plt



#
# LOAD THE CHARACTERISTIC SCALES FROM DISK
# 
characteristic_scales = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt").unsqueeze(2).repeat(1, 1, 100)




class AblationStudyResiduals:
    def __init__(self):

        #
        # LOAD THE ABLATION STUDY LOSS TERMS FROM DISK
        # AND COMPUTE NORMALISED RESIDUALS
        # 

        # Load the residuals from disk and merge them into one tensor
        self.evaluation_loss_terms = torch.zeros(500, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters
    
        for ablation_start in range(0, 500, 100):
            for i in range(ablation_start, ablation_start+100, 10):
                self.evaluation_loss_terms[i:(i+10)] = torch.load(f"./ablation_study_evaluation/eval_residuals/ablation {ablation_start}-{ablation_start+100}/sfere_start {i} sfere_end {i+10}.pt")
    
        # Compute residuals
        self.evaluation_residuals = torch.zeros(500, 4, 100)
    
        # Index 0 - h residual
        self.evaluation_residuals[:, 0, :] = self.evaluation_loss_terms[:, 0, :] / characteristic_scales[:, 0, :]
    
        # Index 1 - uv residual
        self.evaluation_residuals[:, 1, :] = (self.evaluation_loss_terms[:, 1 ,:] + self.evaluation_loss_terms[:, 2, :]) / (characteristic_scales[:, 1, :] + characteristic_scales[:, 2, :])
    
        # Index 2 - closed boundary residual
        self.evaluation_residuals[:, 2, :] = self.evaluation_loss_terms[:, 3, :] / characteristic_scales[:, 3, :]
        
        # Index 3 - closed boundary residual
        self.evaluation_residuals[:, 3, :] = self.evaluation_loss_terms[:, 4, :] / characteristic_scales[:, 4, :]
    
        # Compute aggregate evaluation residual per landscape
        self.evaluation_residuals = torch.mean(self.evaluation_residuals, dim=2)



class HybridHydroPINNResiduals:
    def __init__(self):
        # Load the residuals from disk and merge them into one tensor
        self.evaluation_loss_terms = torch.zeros(500, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters
    
        for i in range(0, 500, 10):
            self.evaluation_loss_terms[i:(i+10)] = torch.load(f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt")
    
        # Compute residuals
        self.evaluation_residuals = torch.zeros(500, 4, 100)
    
        # Index 0 - h residual
        self.evaluation_residuals[:, 0, :] = self.evaluation_loss_terms[:, 0, :] / characteristic_scales[:, 0, :]
    
        # Index 1 - uv residual
        self.evaluation_residuals[:, 1, :] = (self.evaluation_loss_terms[:, 1 ,:] + self.evaluation_loss_terms[:, 2, :]) / (characteristic_scales[:, 1, :] + characteristic_scales[:, 2, :])
    
        # Index 2 - closed boundary residual
        self.evaluation_residuals[:, 2, :] = self.evaluation_loss_terms[:, 3, :] / characteristic_scales[:, 3, :]
        
        # Index 3 - closed boundary residual
        self.evaluation_residuals[:, 3, :] = self.evaluation_loss_terms[:, 4, :] / characteristic_scales[:, 4, :]
    
        # Compute aggregate evaluation residual per landscape
        self.evaluation_residuals = torch.mean(self.evaluation_residuals, dim=2)



class SaltmarshHydroPINNResiduals:
    def __init__(self):
        # Load the residuals from disk and merge them into one tensor
        self.evaluation_loss_terms = torch.zeros(500, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters
    
        for i in range(0, 500, 10):
            self.evaluation_loss_terms[i:(i+10)] = torch.load(f"./Saltmarsh component Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt")
    
        # Compute residuals
        self.evaluation_residuals = torch.zeros(500, 4, 100)
    
        # Index 0 - h residual
        self.evaluation_residuals[:, 0, :] = self.evaluation_loss_terms[:, 0, :] / characteristic_scales[:, 0, :]
    
        # Index 1 - uv residual
        self.evaluation_residuals[:, 1, :] = (self.evaluation_loss_terms[:, 1 ,:] + self.evaluation_loss_terms[:, 2, :]) / (characteristic_scales[:, 1, :] + characteristic_scales[:, 2, :])
    
        # Index 2 - closed boundary residual
        self.evaluation_residuals[:, 2, :] = self.evaluation_loss_terms[:, 3, :] / characteristic_scales[:, 3, :]
        
        # Index 3 - closed boundary residual
        self.evaluation_residuals[:, 3, :] = self.evaluation_loss_terms[:, 4, :] / characteristic_scales[:, 4, :]
    
        # Compute aggregate evaluation residual per landscape
        self.evaluation_residuals = torch.mean(self.evaluation_residuals, dim=2)





def main():

    #
    # LOAD ALL RESIDUAL DATA FROM DISK
    # 

    ablation_study = AblationStudyResiduals()
    hybrid_pinn = HybridHydroPINNResiduals()
    saltmarsh_pinn = SaltmarshHydroPINNResiduals()

    
    #
    # PLOT WITH CONSISTENT COLOURS AND PATTERNINGS
    # 

    plt.figure(figsize=(7, 4))

    # Line-styles
    linestyles = [
        "solid",
        (5, (10, 3)),
        (0, (3, 2))
    ]

    # Color tints
    blue = [
        "#1A3F79",
        "#2C61B2",
        "#4284ED"
    ]
    
    orange = [
        "#934F10",
        "#C76E1A",
        "#FC8E2D"
    ]

    green = [
        "#255B0C",
        "#388216",
        "#4cab20"
    ]
    
    red = [
        "#631414",
        "#9C2525",
        "#D93838"
    ]

    # Select which quantity to plot
    quantity = "uv"

    if quantity == "h":
        plt.title(r"Residual $\mathcal{R}_h$ per Landscape, averaged over 1000 iterations")
        plt.ylabel(r"Residual $\mathcal{R}_h$")
        plt.semilogy(ablation_study.evaluation_residuals[:, 0], label=r"Static Grouped Training", color=blue[0], linestyle=linestyles[0])
        plt.semilogy(hybrid_pinn.evaluation_residuals[:, 0], label=r"Hybrid Training", color=blue[1], linestyle=linestyles[1])
        plt.semilogy(saltmarsh_pinn.evaluation_residuals[:, 0], label=r"PINN Ensemble Training", color=blue[2], linestyle=linestyles[2])
        
    if quantity == "uv":
        plt.title(r"Residual $\mathcal{R}_{uv}$ per Landscape, averaged over 1000 iterations")
        plt.ylabel(r"Residual $\mathcal{R}_{uv}$")
        plt.semilogy(ablation_study.evaluation_residuals[:, 1], label=r"Static Grouped Training", color=orange[0], linestyle=linestyles[0])
        plt.semilogy(hybrid_pinn.evaluation_residuals[:, 1], label=r"Hybrid Training", color=orange[1], linestyle=linestyles[1])
        plt.semilogy(saltmarsh_pinn.evaluation_residuals[:, 1], label=r"PINN Ensemble Training", color=orange[2], linestyle=linestyles[2])
        
    if quantity == "closed_bound":
        plt.title(r"Residual $\mathcal{R}_{bound,closed}$ per Landscape, averaged over 1000 iterations")
        plt.ylabel(r"Residual $\mathcal{R}_{bound,closed}$")
        plt.semilogy(ablation_study.evaluation_residuals[:, 2], label=r"Static Grouped Training", color=green[0], linestyle=linestyles[0])
        plt.semilogy(hybrid_pinn.evaluation_residuals[:, 2], label=r"Hybrid Training", color=green[1], linestyle=linestyles[1])
        plt.semilogy(saltmarsh_pinn.evaluation_residuals[:, 2], label=r"PINN Ensemble Training", color=green[2], linestyle=linestyles[2])
        
    if quantity == "open_bound":
        plt.title(r"Residual $\mathcal{R}_{bound,open}$ per Landscape, averaged over 1000 iterations")
        plt.ylabel(r"Residual $\mathcal{R}_{bound,open}$")
        plt.semilogy(ablation_study.evaluation_residuals[:, 3], label=r"Static Grouped Training", color=red[0], linestyle=linestyles[0])
        plt.semilogy(hybrid_pinn.evaluation_residuals[:, 3], label=r"Hybrid Training", color=red[1], linestyle=linestyles[1])
        plt.semilogy(saltmarsh_pinn.evaluation_residuals[:, 3], label=r"PINN Ensemble Training", color=red[2], linestyle=linestyles[2])
        


    plt.xlabel("Landscape index $n$, sampled at $k(n)$ iterations of SFERE")
    plt.legend(loc="upper right", ncols=1)

    os.makedirs(f"./Residual comparison figures/figures", exist_ok=True)
    plt.savefig(f"./Residual comparison figures/figures/Residual per landscape {quantity}.jpg", dpi=150)

    plt.show()



if __name__ == "__main__":
    main()