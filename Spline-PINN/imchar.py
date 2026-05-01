import torch
import numpy as np
import matplotlib.pyplot as plt





def characterise(numerical_output=10_000):

    # Load reference images from disk
    h = torch.load(f"numerical_output/{numerical_output}/h.pt")
    u = torch.load(f"numerical_output/{numerical_output}/u.pt")
    v = torch.load(f"numerical_output/{numerical_output}/v.pt")
    s = torch.load(f"numerical_output/{numerical_output}/s.pt")
    b = torch.load(f"numerical_output/{numerical_output}/b.pt")

    flow_velocity = torch.sqrt(torch.pow(u, 2) + torch.pow(v, 2))

    # Compute the mean and std.dev of each quantity
    return {
        "stage": numerical_output,
        "mean_h": torch.mean(h).item(),
        "std_h": torch.std(h).item(),
        "min_h": torch.min(h).item(),
        "max_h": torch.max(h).item(),
        "variability_h": (torch.sum(torch.abs(torch.diff(h, dim=3))) + torch.sum(torch.abs(torch.diff(h, dim=2)))).item(),

        "mean_u": torch.mean(u).item(),
        "std_u": torch.std(u).item(),
        "min_u": torch.min(u).item(),
        "max_u": torch.max(u).item(),
        "variability_u": (torch.sum(torch.abs(torch.diff(u, dim=3))) + torch.sum(torch.abs(torch.diff(u, dim=2)))).item(),

        "mean_v": torch.mean(v).item(),
        "std_v": torch.std(v).item(),
        "min_v": torch.min(v).item(),
        "max_v": torch.max(v).item(),
        "variability_v": (torch.sum(torch.abs(torch.diff(v, dim=3))) + torch.sum(torch.abs(torch.diff(v, dim=2)))).item(),

        "mean_s": torch.mean(s).item(),
        "std_s": torch.std(s).item(),
        "min_s": torch.min(s).item(),
        "max_s": torch.max(s).item(),
        "variability_s": (torch.sum(torch.abs(torch.diff(s, dim=3))) + torch.sum(torch.abs(torch.diff(s, dim=2)))).item(),

        "mean_b": torch.mean(b).item(),
        "std_b": torch.std(b).item(),
        "min_b": torch.min(b).item(),
        "max_b": torch.max(b).item(),
        "variability_b": (torch.sum(torch.abs(torch.diff(b, dim=3))) + torch.sum(torch.abs(torch.diff(b, dim=2)))).item(),

        "mean_flow_velocity": torch.mean(flow_velocity).item(),
        "std_flow_velocity": torch.std(flow_velocity).item(),
        "min_flow_velocity": torch.min(flow_velocity).item(),
        "max_flow_velocity": torch.max(flow_velocity).item(),
        "variability_flow_velocity": (torch.sum(torch.abs(torch.diff(flow_velocity, dim=3))) + torch.sum(torch.abs(torch.diff(flow_velocity, dim=2)))).item()
    }


def extract_all(characteristics, name):
    return np.array([output[name] for output in characteristics])





def main():

    output_selection_range = (10_000, 1_000_000)


    # Collect the characteristics of each stage
    characteristics = []

    for output in range(output_selection_range[0], output_selection_range[1]+1, 10_000):
        characteristics.append(characterise(output))

    

    # Plot the characteristics
    xs = np.arange(output_selection_range[0], output_selection_range[1]+1, 10_000)

    # Subplots
    figure, axs = plt.subplots(8, 1, figsize=(20, 11))

    mean_hs = extract_all(characteristics, "mean_h")
    min_hs = extract_all(characteristics, "min_h")
    max_hs = extract_all(characteristics, "max_h")
    axs[0].semilogx(xs, mean_hs, color="tab:blue")
    axs[0].fill_between(xs, min_hs, max_hs, alpha=0.3, facecolor="tab:blue")
    axs[0].set(title="[h] Water layer thickness: min - mean - max")

    variability_hs = extract_all(characteristics, "variability_h")
    axs[1].semilogx(xs, variability_hs, color="tab:blue")
    axs[1].set(title="[h] Water layer thickness: total variability")

    mean_flow_velocitys = extract_all(characteristics, "mean_flow_velocity")
    min_flow_velocitys = extract_all(characteristics, "min_flow_velocity")
    max_flow_velocitys = extract_all(characteristics, "max_flow_velocity")
    axs[2].semilogx(xs, mean_flow_velocitys, color="tab:orange")
    axs[2].fill_between(xs, min_flow_velocitys, max_flow_velocitys, alpha=0.3, facecolor="tab:orange")
    axs[2].set(title="[√(u^2 + v^2)] Flow velocity magnitude: min - mean - max]")

    variability_flow_velocitys = extract_all(characteristics, "variability_flow_velocity")
    axs[3].semilogx(xs, variability_flow_velocitys, color="tab:orange")
    axs[3].set(title="[√(u^2 + v^2)] Flow velocity magnitude: total variability")

    mean_ss = extract_all(characteristics, "mean_s")
    min_ss = extract_all(characteristics, "min_s")
    max_ss = extract_all(characteristics, "max_s")
    axs[4].semilogx(xs, mean_ss, color="tab:brown")
    axs[4].fill_between(xs, min_ss, max_ss, alpha=0.3, facecolor="tab:brown")
    axs[4].set(title="[S] Sedimentary bed elevation: min - mean - max]")

    variability_ss = extract_all(characteristics, "variability_s")
    axs[5].semilogx(xs, variability_ss, color="tab:brown")
    axs[5].set(title="[S] Sedimentary bed elevation: total variability")

    mean_bs = extract_all(characteristics, "mean_b")
    min_bs = extract_all(characteristics, "min_b")
    max_bs = extract_all(characteristics, "max_b")
    axs[6].semilogx(xs, mean_bs, color="tab:green")
    axs[6].fill_between(xs, min_bs, max_bs, alpha=0.3, facecolor="tab:green")
    axs[6].set(title="[B] Vegetation stem density: min - mean - max]")

    variability_bs = extract_all(characteristics, "variability_b")
    axs[7].semilogx(xs, variability_bs, color="tab:green")
    axs[7].set(title="[B] Vegetation stem density: total variability", xlabel="Simulation time (iterations) [LOG-SCALE]")

    figure.tight_layout()


    plt.show()


if __name__ == "__main__":
    main()