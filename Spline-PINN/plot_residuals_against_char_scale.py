import torch
import os
import matplotlib.pyplot as plt




def main():

    # Load the characteristic scales from disk
    characteristic_scales = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt")
    
    # Load the residuals from disk and merge them into one tensor
    evaluation_residuals = torch.zeros(500, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters

    for ablation_start in range(0, 500, 100):
        for i in range(ablation_start, ablation_start+100, 10):
            evaluation_residuals[i:(i+10)] = torch.load(f"./ablation_study_evaluation/eval_residuals/ablation {ablation_start}-{ablation_start+100}/sfere_start {i} sfere_end {i+10}.pt")
            print(torch.load(f"./ablation_study_evaluation/eval_residuals/ablation {ablation_start}-{ablation_start+100}/sfere_start {i} sfere_end {i+10}.pt").shape)

    # Compute aggregate evaluation residual per landscape
    evaluation_residuals = torch.mean(evaluation_residuals, dim=2)

    # Merge the two momentum channels (direction distinction has been removed by evaluating across all landscape orientations)
    vel_residuals = torch.mean(evaluation_residuals[:, [1,2]], dim=1)

    plt.figure(figsize=(7, 4))

    # plt.scatter(characteristic_scales[:, 0], evaluation_residuals[:, 0], label="$S_h$")
    # plt.scatter(characteristic_scales[:, 1], vel_residuals[:], label="$S_{uv}$")
    # plt.scatter(characteristic_scales[:, 3], evaluation_residuals[:, 3], label="$S_{bound,closed}$")
    plt.scatter(characteristic_scales[:, 4], evaluation_residuals[:, 4], label="$S_{bound,open}$")


    plt.title(r"Characteristic Scales per Landscape")


    plt.xlabel("Landscape Characteristic Scale")
    plt.ylabel("Residual")
    plt.legend(loc="upper right", ncols=1)

    # os.makedirs(f"./ablation_study_evaluation/figures", exist_ok=True)
    # plt.savefig(f"./ablation_study_evaluation/figures/Characteristic scales.jpg", dpi=150)

    plt.show()



if __name__ == "__main__":
    main()