import torch
import os
import matplotlib.pyplot as plt




def main():

    characteristic_scales = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt")

    print(characteristic_scales.shape)

    # Compute velocity characteristic scale S_uv
    vel_characteristic_scales = torch.mean(characteristic_scales[:, [1,2]], dim=1)

    print(vel_characteristic_scales.shape)

    plt.figure(figsize=(7, 4))

    plt.plot(characteristic_scales[:, 0] * 100, label=r"$S_h \times 10^2$")
    plt.plot(vel_characteristic_scales, label=r"$\frac{1}{2} (S_u + S_v)$")
    plt.plot(characteristic_scales[:, 3], label=r"$S_{bound,closed}$")
    plt.plot(characteristic_scales[:, 4], label=r"$S_{bound,open}$")


    plt.title(r"Characteristic Scales per Landscape")


    plt.xlabel("Landscape index $n$, sampled at $k(n)$ iterations of SFERE")
    plt.ylabel("Characteristic Scale")
    plt.legend(loc="upper left", ncols=1)

    os.makedirs(f"./ablation_study_evaluation/figures", exist_ok=True)
    plt.savefig(f"./ablation_study_evaluation/figures/Characteristic scales.jpg", dpi=150)

    plt.show()



if __name__ == "__main__":
    main()