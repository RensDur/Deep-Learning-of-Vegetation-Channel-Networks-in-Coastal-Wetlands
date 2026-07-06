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
import multiprocessing
from natsort import natsorted
import psutil



class FitNet(nn.Module):

    def __init__(self, out_channels, hidden_size=16, output_scalar=10):
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
        self.conv1 = nn.Conv2d(1, self.hidden_size, kernel_size=9, padding=4)
        # self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=9, padding=4)
        # self.conv3 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=9, padding=4)

        # Downsampling layers
        # self.down1 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=9, padding=4)  # Maintain resolution, capture large-distance influences
        # self.down2 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=9, padding=4)  # Maintain resolution, capture large-distance influences
        self.down3 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=4, stride=4, padding=0) # Downsample to /4 times the original dimensions
        self.down4 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=2, padding=0)
        self.down5 = nn.Conv2d(self.hidden_size, out_channels, kernel_size=3, padding=1)

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
        # x = self.conv2(x)
        # x = torch.relu(x)
        # x = self.conv3(x)
        # x = torch.relu(x)
        
        # Downsampling layers
        # x = self.down1(x)
        # x = torch.relu(x)
        # x = self.down2(x)
        # x = torch.relu(x)
        x = self.down3(x)
        x = torch.relu(x)
        x = self.down4(x)
        x = torch.relu(x)
        x = self.down5(x)


        out = self.output_scalar * torch.tanh(x / self.output_scalar)

        return out


class CompoundFitNet(nn.Module):

    def __init__(self, spline_variables, torch_device):
        super(CompoundFitNet, self).__init__()

        self.spline_variables = spline_variables

        self.img_channels = len(self.spline_variables) # Number of channels in the input image

        # For each image channel, we dedicate a separate FitNet
        self.nets: list[FitNet] = [FitNet(self.spline_variables[i].hidden_size()) for i in range(len(self.spline_variables))]

        # Vegetation requires a larger output scalar
        self.nets[-1].output_scalar = 2000

        # By default we assume the fitnet to be loaded on the indicated device
        self.device = torch_device

        # And we call the 'to' routine to ensure the position of the nets on the GPU
        self.to(self.device)

    def load_state_from(self, path):

        # This load-state routine is largely inspired by the Logger
        def __load_fitnet_state(path,model,name):

            path = f"{path}/states/{name}.state"
            state = torch.load(path, map_location=self.device)

            if type(model) is not list:
                model = [model]
            for i,m in enumerate(model):
                m.load_state_dict(state['model{}'.format(i)])
        
        __load_fitnet_state(path, self.nets[0], "h")
        __load_fitnet_state(path, self.nets[1], "u")
        __load_fitnet_state(path, self.nets[2], "v")
        __load_fitnet_state(path, self.nets[3], "s")
        __load_fitnet_state(path, self.nets[4], "b")


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




class FitDataset:

    def __init__(self, width, height, device=torch.device("cpu")):

        # Dimensions
        self.width = width
        self.height = height

        # Torch device
        self.device = device

        # Variables in this dataset
        self.variables = SplineArray(
            SplineVariable("h", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("u", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("v", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("s", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("b", 1, requires_derivative=True, requires_laplacian=True),
            device=self.device
        )

        # Load numerical output images from disk
        selected_outputs = [i for i in range(10_000, 1_200_000+1, 10_000)]
        self.dataset_size = len(selected_outputs)
        self.batch_size = 100

        self.numerical_output_states = torch.zeros(
            self.dataset_size,
            5,
            self.width,
            self.height
        )

        for i, st in enumerate(selected_outputs):
            output_h = torch.load(f"numerical_output/{st}/h.pt")
            output_u = torch.load(f"numerical_output/{st}/u.pt")
            output_v = torch.load(f"numerical_output/{st}/v.pt")
            output_s = torch.load(f"numerical_output/{st}/s.pt")
            output_b = torch.load(f"numerical_output/{st}/b.pt")

            composite_image = torch.cat([output_h, output_u, output_v, output_s, output_b], dim=1)

            self.numerical_output_states[i] = composite_image
        
    def ask(self, index=None):

        asked_indices = np.random.choice(self.dataset_size, self.batch_size) if index is None else [index]

        return self.numerical_output_states[asked_indices].to(self.device)


    # def interpolate_states(self, hidden_state, offset):
    #     """
    #     :old_hidden_states: old hidden states (size: bs x (v_size+p_size) x w x h)
    #     :new_hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
    #     :offset: offset in x / y / t direction (vector of size 3 containing values between 0 and 1)
    #     :return: interpolated fields for:
    #         :z: z field
    #         :grad(z): gradient of z field
    #         :laplace(z): laplacian of z field
    #         :dz/dt: velocity of z field
    #         :dz^2/dt^2: acceleration of z field
    #     """

    #     # z field: requires first derivative
    #     h, grad_h, _ = self.variables["h"].interpolate_at(self.variables.extract_from(hidden_state, "h"), offset)

    #     # u field: requires first derivative + laplace
    #     u, grad_u, _ = self.variables["u"].interpolate_at(self.variables.extract_from(hidden_state, "u"), offset)

    #     # v field: requires first derivative + laplace
    #     v, grad_v, _ = self.variables["v"].interpolate_at(self.variables.extract_from(hidden_state, "v"), offset)

    #     # s field: requires first derivative
    #     s, grad_s, _ = self.variables["s"].interpolate_at(self.variables.extract_from(hidden_state, "s"), offset)

    #     # b field: requires first derivative
    #     b, grad_b, _ = self.variables["b"].interpolate_at(self.variables.extract_from(hidden_state, "b"), offset)

    #     return h, u, v, s, b

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



def training_routine(torch_device):

    dataset = FitDataset(800, 800, torch_device)

    net = CompoundFitNet(dataset.variables, torch_device)

    # Create a folder for the hidden state output
    output_folder = f"imfit_output/{dataset.variables.summary()}"
    os.makedirs(f"{output_folder}",exist_ok=True)
    os.makedirs(f"{output_folder}/states",exist_ok=True)

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

    last_saved_state = 0

    def save_state(model, optimizer, name):
        path = f"{output_folder}/states/{name}.state"
        state = {}

        if type(model)is not list:
            model = [model]
        for i,m in enumerate(model):
            state.update({'model{}'.format(i):m.state_dict()})

        if type(optimizer) is not list:
            optimizer = [optimizer]
        for i,o in enumerate(optimizer):
            state.update({'optimizer{}'.format(i):o.state_dict()})

        torch.save(state, path)

    # Enable training
    net.train()

    # Optimizer
    optimizer = Adam(net.parameters(), lr=0.001)
    optimizer = PCGrad(optimizer)


    # Loss function
    def __loss_function(x):
        return torch.pow(x, 2)

    n_epochs = 1000
    n_batches_per_epoch = 100

    # TRAINING LOOP
    for epoch in range(n_epochs):
        for i in range(n_batches_per_epoch):

            batch = dataset.ask()

            predicted_hidden_states = net(batch)

            h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = dataset.interpolate_superres(predicted_hidden_states, 4)

            loss_h = torch.mean(__loss_function(h - batch[:, 0:1]))
            loss_u = torch.mean(__loss_function(u - batch[:, 1:2]))
            loss_v = torch.mean(__loss_function(v - batch[:, 2:3]))
            loss_s = torch.mean(__loss_function(s - batch[:, 3:4]))
            loss_b = torch.mean(__loss_function((b - batch[:, 4:5]) / 1500))

            # Apply log loss
            loss_h = torch.log(loss_h + 0.0001)
            loss_u = torch.log(loss_u + 0.0001)
            loss_v = torch.log(loss_v + 0.0001)
            loss_s = torch.log(loss_s + 0.0001)
            loss_b = torch.log(loss_b + 0.0001)

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

            # Set network grads to zero
            net.zero_grad()

            # PCGrad backprop
            optimizer.pc_backward(pcgrad_losses)

            # Optimization step
            optimizer.step()

            ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) # MB
            ram_usage = round(ram_usage, 2)

            max_vram_allocated = None
            max_vram_reserved = None

            if torch.cuda.is_available():
                max_vram_allocated = torch.cuda.memory.max_memory_allocated() / (1024 * 1024)
                max_vram_reserved = torch.cuda.memory.max_memory_reserved() / (1024 * 1024)

                max_vram_allocated = round(max_vram_allocated, 2)
                max_vram_reserved = round(max_vram_reserved, 2)

            print(f"Epoch {epoch}/{n_epochs} | Batch {i}/{n_batches_per_epoch} >>> Loss {loss:.5f} \t RAM: {ram_usage}MB \t vRAM: {max_vram_allocated}/{max_vram_reserved} (MAX. allocated/reserved, MB)")

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

        # After each epoch, store the network state
        save_state(net.nets[0], optimizer, "h")
        save_state(net.nets[1], optimizer, "u")
        save_state(net.nets[2], optimizer, "v")
        save_state(net.nets[3], optimizer, "s")
        save_state(net.nets[4], optimizer, "b")
        last_saved_state += 1



def evaluation_routine(torch_device):

    selected_num_output = 0

    dataset = FitDataset(800, 800, torch_device)
    net = CompoundFitNet(dataset.variables, torch_device)

    # Load the fitnet state from disk
    output_folder = f"imfit_output/{dataset.variables.summary()}"
    net.load_state_from(output_folder)

    # Load training loss
    # Load loss progression over time
    loss_h_df = pd.read_csv(f"./{output_folder}/loss_h.txt")
    loss_u_df = pd.read_csv(f"./{output_folder}/loss_u.txt")
    loss_v_df = pd.read_csv(f"./{output_folder}/loss_v.txt")
    loss_s_df = pd.read_csv(f"./{output_folder}/loss_s.txt")
    loss_b_df = pd.read_csv(f"./{output_folder}/loss_b.txt")

    losses_df = pd.concat([loss_h_df, loss_u_df, loss_v_df, loss_s_df, loss_b_df], axis=1) 
    

    #
    # EVALUATION
    #

    net.eval()

    #
    # Setup visualisation
    #

    # Plot domain (first time)
    plt.ion()

    # Create subplots
    figure, axs = plt.subplots(2, 3, figsize=(20, 10))

    # Initial black image
    initial_img = torch.zeros(1, 1, 800, 800)

    water_plot = axs[0, 0].imshow(initial_img[0,0].clone().detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=0.1)
    momentum_u_plot = axs[0, 1].imshow(initial_img[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.3, vmax=0.3)
    momentum_v_plot = axs[0, 2].imshow(initial_img[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.2, vmax=0.2)
    sediment_plot = axs[1, 0].imshow(initial_img[0,0].clone().detach().cpu().numpy(), cmap="gray", vmin=0, vmax=0.3)
    vegetation_plot = axs[1, 1].imshow(initial_img[0,0].clone().detach().cpu().numpy(), cmap="YlGn", vmin=0, vmax=1500)
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

    global running
    running = True
    
    def __on_figure_close(event):
        global running
        running = False

    figure.canvas.mpl_connect('close_event', __on_figure_close)

    while running:

        # Ask for the selected numerical output from the dataset
        numerical_output = dataset.ask(index=selected_num_output)

        # Obtain the corresponding hidden state from the CNN
        hidden_state = net(numerical_output)

        # Update the visuals
        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = dataset.interpolate_superres(hidden_state, resolution_factor)

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

        selected_num_output = (selected_num_output + 1) % 100


def main():

    # Find the number of available CPUs, capped at 8
    NUM_CPUS = min(multiprocessing.cpu_count(), 8)
    torch.set_num_threads(NUM_CPUS)
    print(f"Using {NUM_CPUS} threads")
    
    # GPU acceleration
    torch_device = torch.device("cpu")

    if not '--nogpu' in sys.argv:
        if torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        elif torch.cuda.is_available():
            torch_device = torch.device("cuda")

    print(f"Using torch device {torch_device}")

    # Initialize randomization seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # training_routine(torch_device)
    evaluation_routine(torch_device)




if __name__ == "__main__":
    main()