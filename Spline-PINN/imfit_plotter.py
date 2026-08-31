import os
import sys
import torch
import numpy as np
from imfit_general import CompoundFitNet, FitDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec





def main(dataset, snapshot_id):

    torch_device = torch.device("cpu")

    training_dataset = FitDataset(800, 800, torch_device)

    imfit_net = CompoundFitNet(training_dataset.variables, torch_device)
    imfit_net.load_state_from(f"imfit_output/{training_dataset.variables.summary()}") # Immediately load the pre-trained state from disk
    imfit_net.eval()

    # Load the requested snapshot
    if dataset == "train":

        sfere_output = training_dataset.numerical_output_states[snapshot_id:snapshot_id+1]

        imfit_output = imfit_net(sfere_output)

        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = training_dataset.interpolate_superres(imfit_output, resolution_factor=4)

    elif dataset == "verify":

        sfere_output = torch.cat([
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/h.pt"),
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/u.pt"),
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/v.pt"),
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/s.pt"),
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/b.pt"),
        ],dim=1)

        imfit_output = imfit_net(sfere_output)

        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = training_dataset.interpolate_superres(imfit_output, resolution_factor=4)

    elif dataset == "benchmark":

        sfere_output = torch.zeros(1, 5, 800, 800)

        for x in range(800):
            sfere_output[:, :, x:(x+1), :] = torch.sin(torch.pow(torch.Tensor([x]), 1.5) / 200).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 5, 1, 800)

        imfit_output = imfit_net(sfere_output)

        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = training_dataset.interpolate_superres(imfit_output, resolution_factor=4)

    # Store the vegetation profile
    # torch.save(b.detach().cpu(), f"./imfit_vegetation_ic.pt")
    # exit()

    #
    # Compute log loss between the images
    #

    loss_h = torch.pow(h[0, 0].detach().cpu() - sfere_output[0, 0], 2)
    loss_u = torch.pow(u[0, 0].detach().cpu() - sfere_output[0, 1], 2)
    loss_v = torch.pow(v[0, 0].detach().cpu() - sfere_output[0, 2], 2)
    loss_s = torch.pow(s[0, 0].detach().cpu() - sfere_output[0, 3], 2)
    loss_b = torch.pow(b[0, 0].detach().cpu() - sfere_output[0, 4], 2)

    def __norm(loss_image):
        loss_image = loss_image - torch.min(loss_image)
        loss_image = loss_image / torch.max(loss_image)
        loss_image = torch.log(loss_image)
        return loss_image

    loss_h = __norm(loss_h)
    loss_u = __norm(loss_u)
    loss_v = __norm(loss_v)
    loss_s = __norm(loss_s)
    loss_b = __norm(loss_b)

    #
    # Setup visualisation
    #

    # Create subplots
    figure, axs = plt.subplots(3, 5, figsize=(20, 10))

    # Disable ticks
    for i in range(3):
        for j in range(5):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    # Upper row (originals)
    sfere_plot_h = axs[0, 0].imshow(sfere_output[0, 0], cmap="Blues", vmin=0, vmax=0.1)
    sfere_plot_u = axs[0, 1].imshow(sfere_output[0, 1], cmap="bwr", vmin=-0.2, vmax=0.2)
    sfere_plot_v = axs[0, 2].imshow(sfere_output[0, 2], cmap="bwr", vmin=-0.2, vmax=0.2)
    sfere_plot_s = axs[0, 3].imshow(sfere_output[0, 3], cmap="YlOrBr", vmin=0, vmax=0.3)
    sfere_plot_b = axs[0, 4].imshow(sfere_output[0, 4], cmap="YlGn", vmin=0, vmax=1500)

    # Middel row (processed by imfit)
    imfit_plot_h = axs[1, 0].imshow(h[0, 0].detach().cpu(), cmap="Blues", vmin=0, vmax=0.1)
    imfit_plot_u = axs[1, 1].imshow(u[0, 0].detach().cpu(), cmap="bwr", vmin=-0.2, vmax=0.2)
    imfit_plot_v = axs[1, 2].imshow(v[0, 0].detach().cpu(), cmap="bwr", vmin=-0.2, vmax=0.2)
    imfit_plot_s = axs[1, 3].imshow(s[0, 0].detach().cpu(), cmap="YlOrBr", vmin=0, vmax=0.3)
    imfit_plot_b = axs[1, 4].imshow(b[0, 0].detach().cpu(), cmap="YlGn", vmin=0, vmax=1500)

    # Lower row (log loss footprint)
    loss_plot_h = axs[2, 0].imshow(loss_h, cmap="gray", vmin=-10)
    loss_plot_u = axs[2, 1].imshow(loss_u, cmap="gray", vmin=-10)
    loss_plot_v = axs[2, 2].imshow(loss_v, cmap="gray", vmin=-10)
    loss_plot_s = axs[2, 3].imshow(loss_s, cmap="gray", vmin=-10)
    loss_plot_b = axs[2, 4].imshow(loss_b, cmap="gray", vmin=-10)

    plt.colorbar(sfere_plot_h, ticks=np.linspace(0, 0.1, 6, endpoint=True))
    plt.colorbar(sfere_plot_u, ticks=np.linspace(-0.2, 0.2, 9, endpoint=True))
    plt.colorbar(sfere_plot_v, ticks=np.linspace(-0.2, 0.2, 9, endpoint=True))
    plt.colorbar(sfere_plot_s, ticks=np.linspace(0, 0.3, 7, endpoint=True))
    plt.colorbar(sfere_plot_b, ticks=np.linspace(0, 1500, 7, endpoint=True))
    
    plt.colorbar(imfit_plot_h, ticks=np.linspace(0, 0.1, 6, endpoint=True))
    plt.colorbar(imfit_plot_u, ticks=np.linspace(-0.2, 0.2, 9, endpoint=True))
    plt.colorbar(imfit_plot_v, ticks=np.linspace(-0.2, 0.2, 9, endpoint=True))
    plt.colorbar(imfit_plot_s, ticks=np.linspace(0, 0.3, 7, endpoint=True))
    plt.colorbar(imfit_plot_b, ticks=np.linspace(0, 1500, 7, endpoint=True))


    plt.show()


def export_landscape(dataset, snapshot_id):

    torch_device = torch.device("cpu")

    training_dataset = FitDataset(800, 800, torch_device)

    imfit_net = CompoundFitNet(training_dataset.variables, torch_device)
    imfit_net.load_state_from(f"imfit_output/{training_dataset.variables.summary()}") # Immediately load the pre-trained state from disk
    imfit_net.eval()

    # Load the requested snapshot
    if dataset == "train":

        sfere_output = training_dataset.numerical_output_states[snapshot_id:snapshot_id+1]

        imfit_output = imfit_net(sfere_output)

        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = training_dataset.interpolate_superres(imfit_output, resolution_factor=4)

    elif dataset == "verify":

        sfere_output = torch.cat([
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/h.pt"),
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/u.pt"),
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/v.pt"),
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/s.pt"),
            torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_id}/b.pt"),
        ],dim=1)

        imfit_output = imfit_net(sfere_output)

        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = training_dataset.interpolate_superres(imfit_output, resolution_factor=4)

    elif dataset == "benchmark":

        sfere_output = torch.zeros(1, 5, 800, 800)

        for x in range(800):
            sfere_output[:, :, x:(x+1), :] = torch.sin(torch.pow(torch.Tensor([x]), 1.5) / 200).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 5, 1, 800)

        imfit_output = imfit_net(sfere_output)

        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = training_dataset.interpolate_superres(imfit_output, resolution_factor=4)

    # Store the vegetation profile
    # torch.save(b.detach().cpu(), f"./imfit_vegetation_ic.pt")
    # exit()

    #
    # Compute log loss between the images
    #

    loss_h = torch.pow(h[0, 0].detach().cpu() - sfere_output[0, 0], 2)
    loss_u = torch.pow(u[0, 0].detach().cpu() - sfere_output[0, 1], 2)
    loss_v = torch.pow(v[0, 0].detach().cpu() - sfere_output[0, 2], 2)
    loss_s = torch.pow(s[0, 0].detach().cpu() - sfere_output[0, 3], 2)
    loss_b = torch.pow(b[0, 0].detach().cpu() - sfere_output[0, 4], 2)

    def __norm(loss_image):
        loss_image = loss_image - torch.min(loss_image)
        loss_image = loss_image / torch.max(loss_image)
        loss_image = torch.log(loss_image)
        return loss_image

    loss_h = __norm(loss_h)
    loss_u = __norm(loss_u)
    loss_v = __norm(loss_v)
    loss_s = __norm(loss_s)
    loss_b = __norm(loss_b)

    #
    # Setup visualisation
    #

    fig_width = 3.8
    fig_height = 6.0
    left = 0.2
    right = 0.7
    top = 0.4
    bottom = 0.4
    
    fig = plt.figure(figsize=(fig_width, fig_height))

    width_ratios = [1, 0.08]
    height_ratios = [1, 1]
    grid_spec = GridSpec(2, 2, width_ratios=width_ratios, height_ratios=height_ratios, figure=fig) # Add one for the color bars

    axs = [fig.add_subplot(grid_spec[0, 0]), fig.add_subplot(grid_spec[1, 0])]
    caxs = [fig.add_subplot(grid_spec[0, 1]), fig.add_subplot(grid_spec[1, 1])]
    
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

    # Axes labels
    axs[0].set_ylabel("Sedimentary bed elevation ($S$)")
    axs[1].set_ylabel("Vegetation stem density ($B$)")

    # Disable ticks
    for i in range(2):
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # Display imfit images
    imfit_plot_s = axs[0].imshow(s[0, 0].detach().cpu(), cmap="YlOrBr", vmin=0, vmax=0.3)
    imfit_plot_b = axs[1].imshow(b[0, 0].detach().cpu(), cmap="YlGn", vmin=0, vmax=1500)

    # Add colorbars
    plt.colorbar(imfit_plot_s, cax=caxs[0])
    plt.colorbar(imfit_plot_b, cax=caxs[1], ticks=np.linspace(0, 1500, 7))

    # Save the output figure
    os.makedirs(f"./imfit-images/{dataset}", exist_ok=True)
    fig.savefig(f"./imfit-images/{dataset}/ImFit output {snapshot_id}.jpg", dpi=150)



if __name__ == "__main__":


    if "--export_landscape" in sys.argv:
        export_landscape("verify", 99)
    else:
        main("verify", 99)
