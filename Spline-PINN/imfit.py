import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from itertools import chain
from pcgrad.pcgrad import PCGrad
import numpy as np
import pandas as pd
import math
from spline.spline_variable import SplineVariable
from spline.spline_array import SplineArray
import matplotlib.pyplot as plt




class FitDataset:

    def __init__(self, width, height, device=torch.device("cpu")):

        # Dimensions
        self.width = width
        self.height = height

        # Torch device
        self.device = device

        # Variables in this dataset
        self.variables = SplineArray(
            SplineVariable("h", 1, requires_derivative=True, requires_laplacian=True),                           # h describes the zero-meaned surface height, on top of H0
            SplineVariable("u", 2, requires_derivative=True, requires_laplacian=True),
            SplineVariable("v", 2, requires_derivative=True, requires_laplacian=True),
            SplineVariable("s", 2, requires_derivative=True, requires_laplacian=True),
            SplineVariable("b", 2, requires_derivative=True, requires_laplacian=True),
            device=self.device
        )

        # Hidden state
        self.hidden_state = torch.rand(
            1,
            self.variables.hidden_size(),
            self.width-1,
            self.height-1,
            device=self.device
        )


    def interpolate_states(self, hidden_state, offset):
        """
        :old_hidden_states: old hidden states (size: bs x (v_size+p_size) x w x h)
        :new_hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
        :offset: offset in x / y / t direction (vector of size 3 containing values between 0 and 1)
        :return: interpolated fields for:
            :z: z field
            :grad(z): gradient of z field
            :laplace(z): laplacian of z field
            :dz/dt: velocity of z field
            :dz^2/dt^2: acceleration of z field
        """

        # z field: requires first derivative
        h, grad_h, _ = self.variables["h"].interpolate_at(self.variables.extract_from(hidden_state, "h"), offset)

        # u field: requires first derivative + laplace
        u, grad_u, _ = self.variables["u"].interpolate_at(self.variables.extract_from(hidden_state, "u"), offset)

        # v field: requires first derivative + laplace
        v, grad_v, _ = self.variables["v"].interpolate_at(self.variables.extract_from(hidden_state, "v"), offset)

        # s field: requires first derivative
        s, grad_s, _ = self.variables["s"].interpolate_at(self.variables.extract_from(hidden_state, "s"), offset)

        # b field: requires first derivative
        b, grad_b, _ = self.variables["b"].interpolate_at(self.variables.extract_from(hidden_state, "b"), offset)

        return h, u, v, s, b

    def interpolate_superres(self, hidden_states, resolution_factor):
        """
        :hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
        "resolution_factor": resolution factor for superres interpolation
        :return: interpolated fields for:
            :z: z field
            :grad(z): gradient of z field
            :laplace(z): laplacian of z field
            :dz/dt: velocity of z field
            :dz^2/dt^2: acceleration of z field
        """

        # h field: requires first derivative
        h, grad_h, _ = self.variables["h"].interpolate_superres_at(self.variables.extract_from(hidden_states, "h"), resolution_factor)

        # u field: requires first derivative + laplace
        u, grad_u, _ = self.variables["u"].interpolate_superres_at(self.variables.extract_from(hidden_states, "u"), resolution_factor)

        # v field: requires first derivative + laplace
        v, grad_v, _ = self.variables["v"].interpolate_superres_at(self.variables.extract_from(hidden_states, "v"), resolution_factor)

        # s field: requires first derivative + laplace
        s, grad_s, _ = self.variables["s"].interpolate_superres_at(self.variables.extract_from(hidden_states, "s"), resolution_factor)

        # s field: requires first derivative + laplace
        b, grad_b, _ = self.variables["b"].interpolate_superres_at(self.variables.extract_from(hidden_states, "b"), resolution_factor)

        return h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b



class FitNet(nn.Module):

    def __init__(self, out_channels, hidden_size=32, output_scalar=2):
        """
        :orders_v: order of spline for velocity potential (should be at least 2)
        :orders_p: order of spline for pressure field
        :hidden_size: hidden size of neural net
        :interpolation_size: size of first interpolation layer for v_cond and v_mask
        """
        super(FitNet, self).__init__()

        self.hidden_size = out_channels
        self.out_channels = out_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, self.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_size, self.hidden_size*2, kernel_size=3, padding=1)

        # Downsampling layers
        self.down1 = nn.Conv2d(self.hidden_size*2, self.hidden_size*2, kernel_size=9, padding=4)  # Maintain resolution, capture large-distance influences
        self.down2 = nn.Conv2d(self.hidden_size*2, self.hidden_size*2, kernel_size=4, stride=4, padding=0) # Downsample to /4 times the original dimensions
        self.down3 = nn.Conv2d(self.hidden_size*2, self.hidden_size, kernel_size=2, padding=0)
        self.down4 = nn.Conv2d(self.hidden_size, out_channels, kernel_size=3, padding=1)

        self.output_scalar = output_scalar

    def forward(self, input_image):
        """
        :hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
        :v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
        :v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
        :return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
        """

        x = input_image

        # Convolutional layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        
        # Downsampling layers
        x = self.down1(x)
        x = torch.relu(x)
        x = self.down2(x)
        x = torch.relu(x)
        x = self.down3(x)
        x = torch.relu(x)
        x = self.down4(x)


        out = self.output_scalar * torch.tanh(x / self.output_scalar)

        return out


class CompoundFitNet(nn.Module):

    def __init__(self, spline_variables):
        super(CompoundFitNet, self).__init__()

        self.spline_variables = spline_variables

        self.img_channels = len(self.spline_variables) # Number of channels in the input image

        # For each image channel, we dedicate a separate FitNet
        self.nets: list[FitNet] = [FitNet(self.spline_variables[i].hidden_size()) for i in range(len(self.spline_variables))]

        # Vegetation requires a larger output scalar
        self.nets[-1].output_scalar = 2000

    def to(self, torch_device):
        super(CompoundFitNet, self).to(torch_device)
        self.nets = [n.to(torch_device) for n in self.nets]
        return self

    def train(self):
        [n.train() for n in self.nets]

    def eval(self):
        [n.eval() for n in self.nets]

    def parameters(self):
        return chain.from_iterable(n.parameters() for n in self.nets)

    def forward(self, input_image):
        """
        :hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
        :v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
        :v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
        :return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
        """

        compound = []

        for i in range(self.img_channels):
            compound.append(
                self.nets[i](input_image[:,i:i+1,:,:])
            )

        out = torch.cat(compound, dim=1)

        return out


def training_loop(SELECTED_NUMERICAL_OUTPUT, torch_device):
    
    dataset = FitDataset(200, 200, torch_device)

    net = CompoundFitNet(dataset.variables).to(torch_device)

    # Enable training
    net.train()

    # Optimizer
    optimizer = Adam(net.parameters(), lr=0.0001)
    optimizer = PCGrad(optimizer)

    # Load reference images from disk
    ref_h = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/h.pt").to(torch_device)
    ref_u = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/u.pt").to(torch_device)
    ref_v = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/v.pt").to(torch_device)
    ref_s = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/s.pt").to(torch_device)
    ref_b = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/b.pt").to(torch_device)

    input_image = torch.cat([ref_h, ref_u, ref_v, ref_s, ref_b], dim=1)

    # Create a folder for the hidden state output
    output_folder = f"numerical_spline_converted/{dataset.variables.summary()}/{SELECTED_NUMERICAL_OUTPUT}"
    os.makedirs(f"{output_folder}",exist_ok=True)

    # Create an empty file for every loss component
    with open(f"{output_folder}/loss_h.txt", "w") as file:
        file.write(f"loss_h\n")
    with open(f"{output_folder}/loss_u.txt", "w") as file:
        file.write(f"loss_u\n")
    with open(f"{output_folder}/loss_v.txt", "w") as file:
        file.write(f"loss_v\n")
    with open(f"{output_folder}/loss_s.txt", "w") as file:
        file.write(f"loss_s\n")
    with open(f"{output_folder}/loss_b.txt", "w") as file:
        file.write(f"loss_b\n")
    with open(f"{output_folder}/loss_total.txt", "w") as file:
        file.write(f"loss_total\n")

    # Try to load the latest hidden state
    try:
        dataset.hidden_state = torch.load(f"{output_folder}/hidden_state.pt")
    except:
        print(f"Unable to load previous optimal hidden state from disk")

    # Setup visualisation

    # # Plot domain (first time)
    # plt.ion()

    # # Create subplots
    # figure, axs = plt.subplots(2, 2, figsize=(20, 10))

    # sediment_plot = axs[0, 0].imshow(ref_s[0,0].clone().detach().cpu().numpy(), cmap="gray", vmin=0, vmax=0.2)
    # sediment_plot_under_veg = axs[0, 1].imshow(ref_s[0,0].clone().detach().cpu().numpy(), cmap="gray", vmin=0, vmax=0.2)
    # vegetation_plot = axs[0, 1].imshow(ref_b[0,0].clone().detach().cpu().numpy(), cmap="YlGn", vmin=0, vmax=1500, alpha=0.8)

    # momentum_u_plot = axs[1, 0].imshow(ref_u[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.2, vmax=0.2)
    # momentum_v_plot = axs[1, 1].imshow(ref_v[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.2, vmax=0.2)

    # # setting title
    # axs[0, 0].set(title="Sediment bed", xlabel="Cross shore", ylabel="Along shore")
    # axs[0, 1].set(title="Sediment bed with vegetation", xlabel="Cross shore", ylabel="Along shore")
    # axs[1, 0].set(title="Momentum u (x-direction)", xlabel="Cross shore", ylabel="Along shore")
    # axs[1, 1].set(title="Momentum v (y-direction)", xlabel="Cross shore", ylabel="Along shore")

    # # Color bars
    # plt.colorbar(sediment_plot)
    # plt.colorbar(sediment_plot_under_veg)
    # plt.colorbar(vegetation_plot)
    # plt.colorbar(momentum_u_plot)
    # plt.colorbar(momentum_v_plot)

    # # In interactive mode, plt.show() immediately returns
    # plt.show()

    # Loss function
    def __loss_function(x):
        return torch.pow(x, 2)

    # Training loop
    EPOCHS = 1000
    N_BATCHES = 10
    N_SAMPLES = 50
    resolution_factor = 4

    # Store the minimum loss output
    min_loss = 10

    for epoch in range(EPOCHS):
        for batch in range(N_BATCHES):

            output_hidden_state = net(input_image)

            loss_h = 0
            loss_u = 0
            loss_v = 0
            loss_s = 0
            loss_b = 0

            for i in range(N_SAMPLES):
                # Randomly pick a sampling point
                sample = torch.rand(2)

                y_offset = min(int(resolution_factor*sample[0]),resolution_factor-1)
                x_offset = min(int(resolution_factor*sample[1]),resolution_factor-1)

                sample_h = ref_h[:, :, y_offset::resolution_factor, x_offset::resolution_factor]
                sample_u = ref_u[:, :, y_offset::resolution_factor, x_offset::resolution_factor]
                sample_v = ref_v[:, :, y_offset::resolution_factor, x_offset::resolution_factor]
                sample_s = ref_s[:, :, y_offset::resolution_factor, x_offset::resolution_factor]
                sample_b = ref_b[:, :, y_offset::resolution_factor, x_offset::resolution_factor]

                offset = torch.floor(sample*resolution_factor)/resolution_factor
                offset = offset.to(torch_device)

                # Interpolate states
                h, u, v, s, b = dataset.interpolate_states(output_hidden_state, offset)

                # Compute loss
                loss_h = loss_h + torch.mean(__loss_function(h - sample_h[:,:,1:-1,1:-1]))
                loss_u = loss_u + torch.mean(__loss_function(u - sample_u[:,:,1:-1,1:-1]))
                loss_v = loss_v + torch.mean(__loss_function(v - sample_v[:,:,1:-1,1:-1]))
                loss_s = loss_s + torch.mean(__loss_function(s - sample_s[:,:,1:-1,1:-1]))
                loss_b = loss_b + torch.mean(__loss_function(b - sample_b[:,:,1:-1,1:-1]))

            # Normalize for the number of samples
            loss_h = loss_h / N_SAMPLES
            loss_u = loss_u / N_SAMPLES
            loss_v = loss_v / N_SAMPLES
            loss_s = loss_s / N_SAMPLES
            loss_b = loss_b / N_SAMPLES

            loss_b = loss_b / 1500**2

            # Log loss
            loss_h = torch.log(loss_h + 1e-5)
            loss_u = torch.log(loss_u + 1e-5)
            loss_v = torch.log(loss_v + 1e-5)
            loss_s = torch.log(loss_s + 1e-5)
            loss_b = torch.log(loss_b + 1e-5)

            # Sum loss for stats
            loss = loss_h + loss_u + loss_v + loss_s + loss_b

            # PCGrad losses
            pcgrad_losses = [
                loss_h,
                loss_u,
                loss_v,
                loss_s,
                loss_b,
            ]

            # Backprop
            net.zero_grad()
            
            # PCGrad backprop
            optimizer.pc_backward(pcgrad_losses)

            # Optimization step
            optimizer.step()

            # Put the hidden state back
            dataset.hidden_state = output_hidden_state.detach()

            # Report the progression of loss
            with open(f"{output_folder}/loss_h.txt", "a") as file:
                file.write(f"{loss_h}\n")
            with open(f"{output_folder}/loss_u.txt", "a") as file:
                file.write(f"{loss_u}\n")
            with open(f"{output_folder}/loss_v.txt", "a") as file:
                file.write(f"{loss_v}\n")
            with open(f"{output_folder}/loss_s.txt", "a") as file:
                file.write(f"{loss_s}\n")
            with open(f"{output_folder}/loss_b.txt", "a") as file:
                file.write(f"{loss_b}\n")
            with open(f"{output_folder}/loss_total.txt", "a") as file:
                file.write(f"{loss}\n")

            # If this was the best run until now, save the state
            if loss < min_loss:
                min_loss = loss
                torch.save(dataset.hidden_state.cpu(), f"{output_folder}/hidden_state.pt")

                # Report the loss
                with open(f"{output_folder}/min_loss.txt", "w") as file:
                    file.write(f"Minimum loss stored: {min_loss}")

        # Report loss
        print(f"Loss (epoch {epoch}): {loss}")

        # # Update the visuals
        # h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = dataset.interpolate_superres(output_hidden_state, resolution_factor)

        # sediment_plot.set_data(s[0,0].detach().cpu().numpy())
        # sediment_plot_under_veg.set_data(s[0,0].detach().cpu().numpy())
        # vegetation_plot.set_data(b[0,0].detach().cpu().numpy())

        # momentum_u_plot.set_data(u[0,0].detach().cpu().numpy())
        # momentum_v_plot.set_data(v[0,0].detach().cpu().numpy())

        # # Plot the domain (update existing plot)
        # # Draw updated values
        # figure.canvas.draw()

        # # UI Loop: process all pending UI events
        # figure.canvas.flush_events()


def evaluation_loop(SELECTED_NUMERICAL_OUTPUT, torch_device):
       
    dataset = FitDataset(200, 200, torch_device)

    # Point to the same output folder used during training
    output_folder = f"numerical_spline_converted/{dataset.variables.summary()}/{SELECTED_NUMERICAL_OUTPUT}"

    # Try to load the latest hidden state
    try:
        dataset.hidden_state = torch.load(f"{output_folder}/hidden_state.pt").to(torch_device)
    except:
        print(f"Unable to load previous optimal hidden state from disk")

    # Load reference images from disk
    ref_h = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/h.pt").to(torch_device)
    ref_u = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/u.pt").to(torch_device)
    ref_v = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/v.pt").to(torch_device)
    ref_s = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/s.pt").to(torch_device)
    ref_b = torch.load(f"numerical_output/{SELECTED_NUMERICAL_OUTPUT}/b.pt").to(torch_device)

    input_image = torch.cat([ref_h, ref_u, ref_v, ref_s, ref_b], dim=1)

    # Load loss progression over time
    loss_h_df = pd.read_csv(f"./{output_folder}/loss_h.txt")
    loss_u_df = pd.read_csv(f"./{output_folder}/loss_u.txt")
    loss_v_df = pd.read_csv(f"./{output_folder}/loss_v.txt")
    loss_s_df = pd.read_csv(f"./{output_folder}/loss_s.txt")
    loss_b_df = pd.read_csv(f"./{output_folder}/loss_b.txt")

    losses_df = pd.concat([loss_h_df, loss_u_df, loss_v_df, loss_s_df, loss_b_df], axis=1)    

    # Setup visualisation

    # Plot domain (first time)
    plt.ion()

    # Create subplots
    figure, axs = plt.subplots(2, 3, figsize=(20, 10))

    water_plot = axs[0, 0].imshow(ref_h[0,0].clone().detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=0.02)
    momentum_u_plot = axs[0, 1].imshow(ref_u[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.2, vmax=0.2)
    momentum_v_plot = axs[0, 2].imshow(ref_v[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.2, vmax=0.2)
    sediment_plot = axs[1, 0].imshow(ref_s[0,0].clone().detach().cpu().numpy(), cmap="gray", vmin=0, vmax=0.2)
    vegetation_plot = axs[1, 1].imshow(ref_b[0,0].clone().detach().cpu().numpy(), cmap="YlGn", vmin=0, vmax=1500)
    losses_df.plot(ax=axs[1, 2])


    # setting title
    axs[0, 0].set(title="Water Layer Thickness", xlabel="Cross shore", ylabel="Along shore")
    axs[0, 1].set(title="Momentum u (x-direction)", xlabel="Cross shore", ylabel="Along shore")
    axs[0, 2].set(title="Momentum v (y-direction)", xlabel="Cross shore", ylabel="Along shore")
    axs[1, 0].set(title="Sediment bed", xlabel="Cross shore", ylabel="Along shore")
    axs[1, 1].set(title="Vegetation density", xlabel="Cross shore", ylabel="Along shore")

    # Color bars
    plt.colorbar(water_plot)
    plt.colorbar(momentum_u_plot)
    plt.colorbar(momentum_v_plot)
    plt.colorbar(sediment_plot)
    plt.colorbar(vegetation_plot)

    # In interactive mode, plt.show() immediately returns
    plt.show()

    # Resolution factor
    resolution_factor = 4

    while True:

        # Update the visuals
        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = dataset.interpolate_superres(dataset.hidden_state, resolution_factor)

        # Quick loss calculation
        loss_h = torch.pow(h - ref_h, 2)
        loss_u = torch.pow(u - ref_u, 2)
        loss_v = torch.pow(v - ref_v, 2)
        loss_s = torch.pow(s - ref_s, 2)
        loss_b = torch.pow(b - ref_b, 2)

        water_plot.set_data(h[0,0].detach().cpu().numpy())
        momentum_u_plot.set_data(u[0,0].detach().cpu().numpy())
        momentum_v_plot.set_data(v[0,0].detach().cpu().numpy())
        sediment_plot.set_data(s[0,0].detach().cpu().numpy())
        vegetation_plot.set_data(b[0,0].detach().cpu().numpy())

        # Plot the domain (update existing plot)
        # Draw updated values
        figure.canvas.draw()

        # UI Loop: process all pending UI events
        figure.canvas.flush_events()





if __name__ == "__main__":

    SELECTED_NUMERICAL_OUTPUT = 300000

    
    # Program mode
    mode = "train"
    if '--vis' in sys.argv:
        mode = "eval"
    
    # GPU acceleration
    torch_device = torch.device("cpu")

    if not '--nogpu' in sys.argv:
        if torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        elif torch.cuda.is_available():
            torch_device = torch.device("cuda")

    print(f"Using torch device {torch_device}")


    if mode == "train":
        training_loop(SELECTED_NUMERICAL_OUTPUT, torch_device)
    elif mode == "eval":
        evaluation_loop(SELECTED_NUMERICAL_OUTPUT, torch_device)
    else:
        raise Exception(f"Unrecognized mode: {mode}")