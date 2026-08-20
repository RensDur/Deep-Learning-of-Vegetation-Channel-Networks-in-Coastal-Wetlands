import torch
import os
import matplotlib.pyplot as plt




def main():

    characteristic_scales = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt")

    plt.figure(figsize=(7, 4))

    plt.semilogy(characteristic_scales[:, 0], label="$S_h$")
    plt.semilogy(characteristic_scales[:, 1], label="$S_u$")
    plt.semilogy(characteristic_scales[:, 2], label="$S_v$")
    plt.semilogy(characteristic_scales[:, 3], label="$S_{bound,closed}$")
    plt.semilogy(characteristic_scales[:, 4], label="$S_{bound,open}$")


    plt.title(r"Characteristic Scales per Landscape")


    plt.xlabel("Landscape index $n$, sampled at $k(n)$ iterations of SFERE")
    plt.ylabel("Characteristic Scale [log]")
    plt.legend(loc="lower right", ncols=1)

    os.makedirs(f"./ablation_study_evaluation/figures", exist_ok=True)
    plt.savefig(f"./ablation_study_evaluation/figures/Characteristic scales.jpg", dpi=150)

    # plt.show()



if __name__ == "__main__":
    main()