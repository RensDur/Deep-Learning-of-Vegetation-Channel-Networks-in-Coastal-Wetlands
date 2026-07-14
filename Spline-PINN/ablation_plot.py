import torch
import matplotlib.pyplot as plt
from window import PerformanceSummaryWindow




def main():

    selected_model = '0-100'

    ablation_output = torch.load(f"./ablation_output/ablation {selected_model}/iteration_0.pt").detach().cpu()
    
    # Load all ablation models in the directory
    ablation_output = torch.cat([
        torch.load(f"./ablation_output/ablation {selected_model}/iteration_{i}.pt") for i in range(0, 1000+1, 100)
    ], dim=0)

    # Plot the images in a performance window
    win = PerformanceSummaryWindow(ablation_output.shape[3], ablation_output.shape[2], ablation_output.shape[0], 100)

    win.open()

    for i in range(0, ablation_output.shape[0]):
        win.set_data(
            ablation_output[i, 0],
            ablation_output[i, 1],
            ablation_output[i, 2],
            ablation_output[i, 3],
            ablation_output[i, 4],
            i*100
        )

    while win.is_open:
        win.update()



if __name__ == "__main__":
    main()