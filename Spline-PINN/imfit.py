import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
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
            SplineVariable("u", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("v", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("s", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("b", 1, requires_derivative=True, requires_laplacian=True),
            device=self.device
        )

        # Hidden state
        self.hidden_state = torch.zeros(
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
        b, grad_b, _ = self.variables["s"].interpolate_at(self.variables.extract_from(hidden_state, "s"), offset)

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

    def __init__(self, spline_variables, hidden_size=32):
        """
        :orders_v: order of spline for velocity potential (should be at least 2)
        :orders_p: order of spline for pressure field
        :hidden_size: hidden size of neural net
        :interpolation_size: size of first interpolation layer for v_cond and v_mask
        """
        super(FitNet, self).__init__()

        self.hidden_size = hidden_size
        self.spline_variables = spline_variables

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.spline_variables.hidden_size(), self.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_size, self.spline_variables.hidden_size(), kernel_size=3, padding=1)

    def forward(self, hidden_state):
        """
        :hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
        :v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
        :v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
        :return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
        """

        x = self.conv1(hidden_state)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)

        out = x

        return out


def main():

    torch_device = torch.device("cpu")

    # if torch.backends.mps.is_available():
    #     torch_device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     torch_device = torch.device("cuda")
    
    dataset = FitDataset(200, 200, torch_device)

    net = FitNet(dataset.variables).to(torch_device)

    # Optimizer
    optimizer = Adam(net.parameters(), lr=0.0001)

    # Enable training
    net.train()

    # Load reference images from disk
    ref_h = torch.ones(1, 1, 800, 800).to(torch_device)
    ref_u = torch.ones(1, 1, 800, 800).to(torch_device)
    ref_v = torch.ones(1, 1, 800, 800).to(torch_device)
    ref_s = torch.ones(1, 1, 800, 800).to(torch_device)
    ref_b = torch.ones(1, 1, 800, 800).to(torch_device)

    # Setup visualisation

    # Plot domain (first time)
    plt.ion()

    # Create subplots
    figure, axs = plt.subplots(2, 2, figsize=(20, 10))

    sediment_plot = axs[0, 0].imshow(ref_s[0,0], cmap="gray", vmin=0, vmax=0.2)
    sediment_plot_under_veg = axs[0, 1].imshow(ref_s[0,0], cmap="gray", vmin=0, vmax=0.2)
    vegetation_plot = axs[0, 1].imshow(ref_b[0,0], cmap="YlGn", vmin=0, vmax=1500, alpha=0.8)

    momentum_u_plot = axs[1, 0].imshow(ref_u[0,0], cmap="bwr", vmin=-0.2, vmax=0.2)
    momentum_v_plot = axs[1, 1].imshow(ref_v[0,0], cmap="bwr", vmin=-0.2, vmax=0.2)

    # setting title
    axs[0, 0].set(title="Sediment bed", xlabel="Cross shore", ylabel="Along shore")
    axs[0, 1].set(title="Sediment bed with vegetation", xlabel="Cross shore", ylabel="Along shore")
    axs[1, 0].set(title="Momentum u (x-direction)", xlabel="Cross shore", ylabel="Along shore")
    axs[1, 1].set(title="Momentum v (y-direction)", xlabel="Cross shore", ylabel="Along shore")

    # Color bars
    plt.colorbar(sediment_plot)
    plt.colorbar(sediment_plot_under_veg)
    plt.colorbar(vegetation_plot)
    plt.colorbar(momentum_u_plot)
    plt.colorbar(momentum_v_plot)

    # In interactive mode, plt.show() immediately returns
    plt.show()

    # Loss function
    def __loss_function(x):
        return torch.pow(x, 2)

    # Training loop
    EPOCHS = 100
    N_SAMPLES = 10
    resolution_factor = 4

    for epoch in range(EPOCHS):

        output_hidden_state = net(dataset.hidden_state)

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

        # Log loss
        loss = torch.log(loss_h + loss_u + loss_v + loss_s + loss_b)

        # Report loss
        print(f"Loss: {loss}")

        # Backprop
        net.zero_grad()
        loss.backward()

        # Optimization step
        optimizer.step()

        # Update the visuals
        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = dataset.interpolate_superres(output_hidden_state, resolution_factor)

        sediment_plot.set_data(s[0,0].detach().cpu().numpy())
        sediment_plot_under_veg.set_data(s[0,0].detach().cpu().numpy())
        vegetation_plot.set_data(b[0,0].detach().cpu().numpy())

        momentum_u_plot.set_data(u[0,0].detach().cpu().numpy())
        momentum_v_plot.set_data(v[0,0].detach().cpu().numpy())

        # Plot the domain (update existing plot)
        # Draw updated values
        figure.canvas.draw()

        # UI Loop: process all pending UI events
        figure.canvas.flush_events()








if __name__ == "__main__":
    main()