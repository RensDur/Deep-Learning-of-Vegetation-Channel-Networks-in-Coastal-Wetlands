import torch
import torch.nn.functional as F
import numpy as np
import math
from imfit_general import CompoundFitNet
from spline.spline_variable import SplineVariable
from spline.spline_array import SplineArray


class Dataset:
    
    def __init__(self, params, device=torch.device("cpu"), types=None, water_strategies=None, vegetation_ics=None, orientations=None):

        # Local copy of the parameters
        self.params = params

        # Dimensions
        self.width = params.width
        self.height = params.height
        self.resolution_factor = params.resolution_factor
        self.width_fullres = self.resolution_factor * self.width
        self.height_fullres = self.resolution_factor * self.height

        self.padding = 5
        self.padding_fullres = self.padding * self.resolution_factor

        self.dx = params.separation
        self.dy = params.separation

        # Dataset sizes
        self.dataset_size = params.dataset_size
        self.batch_size = params.batch_size

        # Sampling
        self.n_samples = params.n_samples

        # Random reset
        self.average_sequence_length = params.average_sequence_length

        # Torch device
        self.device = device

        # Variables in this dataset
        self.variables = SplineArray(
            SplineVariable("h", 1, requires_derivative=True),                           # h describes the zero-meaned surface height, on top of H0
            SplineVariable("u", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("v", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("s", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("b", 1, requires_derivative=True, requires_laplacian=True),
            device=self.device
        )

        # Hidden state
        self.hidden_states = torch.zeros(
            self.dataset_size,
            self.variables.hidden_size(),
            self.width-1,
            self.height-1
        )

        # Boundary conditions and masking
        self.closed_mask = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.opened_mask = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.h_mask = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.h_cond = torch.zeros(self.dataset_size, 1, self.width, self.height)

        self.closed_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.opened_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.h_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.h_cond_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)

        # Create a CompoundFitNet (which is a compound of 5 Image Fitting CNNs: one for each variable h u v s b)
        self.imfit_net = CompoundFitNet(self.variables, self.device)
        self.imfit_net.load_state_from(f"imfit_output/{self.variables.summary()}") # Immediately load the pre-trained state from disk
        self.imfit_net.eval()

        # Load snapshots from disk onto CPU memory
        self.num_sfere_samples = self.params.sfere_end - self.params.sfere_start
        self.sfere_snapshots = torch.zeros(self.num_sfere_samples, 5, 800, 800).to(torch.device("cpu"))

        for i, snapshot_index in enumerate(range(self.params.sfere_start, self.params.sfere_end)):
            self.sfere_snapshots[i:i+1] = torch.cat([
                torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_index}/h.pt", map_location=torch.device("cpu")),
                torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_index}/u.pt", map_location=torch.device("cpu")),
                torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_index}/v.pt", map_location=torch.device("cpu")),
                torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_index}/s.pt", map_location=torch.device("cpu")),
                torch.load(f"snapshots-log-slowdown/snapshot_{snapshot_index}/b.pt", map_location=torch.device("cpu")),
            ], dim=1)

        # Environment information
        self.types = [
            "numerical-saltmarsh",
            # "topoflat",
            # "toposlope",
            # "toposlope-curved",
            # "toposharp-vegmax",
            # "toposharp-vegslope"

            # "topoflat-closed-oscillator",
            # "topoflat-closed-oscillator-multiple",
            # "topoflat-closed-oscillator-reflection",
            # "toposlope-closed-oscillator",
            # "toposlope-closed-oscillator-multiple",
            # "toposlope-closed-oscillator-reflection",
            # "topoveg-closed-oscillator",
            # "topoveg-closed-oscillator",
            # "topoveg-closed-oscillator"
        ] if types is None else types

        # Water in- and outflow strategies
        self.water_strategies = [
            "Hin",
            # "tidal-flow"
        ] if water_strategies is None else water_strategies

        # Vegetation initial conditions
        self.vegetation_ics = [
            # "uniform-noise",
            # "random-gaussians",
            # "vd-vijsel",
            # "empty",
            "elsewhere-specified"
        ] if vegetation_ics is None else vegetation_ics

        self.orientations = [
            "north",
            "east",
            "south",
            "west"
        ] if orientations is None else orientations

        print(f"Running with types {self.types}, water strategies {self.water_strategies}, vegetation-ics {self.vegetation_ics} and orientations {self.orientations}")
        print(f"Active numerical outputs: {slice(self.params.sfere_start, self.params.sfere_end)}")

        self.env_type = np.random.choice(self.types, self.dataset_size)
        self.env_water_strategy = np.random.choice(self.water_strategies, self.dataset_size)
        self.env_vegetation_ics = np.random.choice(self.vegetation_ics, self.dataset_size)
        self.env_orientation = np.random.choice(self.orientations, self.dataset_size)
        self.env_seed = 2.0 * math.pi * torch.floor(1000 * torch.rand(self.dataset_size))
        self.env_time = torch.zeros(self.dataset_size)

        # Environment resetting
        self.t = 0
        self.i = 0
        self.warmup_t = 0
        self.warmup_reset_at = 1

        # Reset all environments
        print("Resetting all environments (in batches)")

        processed = 0
        batch_size = self.params.batch_size

        while processed < self.dataset_size:
            current_batch = min(batch_size, self.dataset_size - processed)

            self.reset(range(processed, processed + current_batch))
            processed += current_batch

    def hidden_size(self):
        return self.variables.hidden_size()
    
    def group_by_type(self, indices):
        """
        This function outputs a dictionary grouping environments with the same type together
        """

        grouping = {}

        # Initialize groups with empty lists
        for t in self.types:
            grouping[t] = []

        # Group environments
        for i in indices:
            grouping[self.env_type[i]].append(i)

        # Remove any empty groups
        for g in list(grouping):
            if not grouping[g]:
                grouping.pop(g)

        return grouping
    
    def group_by_water_strategy(self, indices):
        """
        This function outputs a dictionary grouping environments with the same water strategy together
        """

        grouping = {}

        # Initialize groups with empty lists
        for t in self.water_strategies:
            grouping[t] = []

        # Group environments
        for i in indices:
            grouping[self.env_water_strategy[i]].append(i)

        # Remove any empty groups
        for g in list(grouping):
            if not grouping[g]:
                grouping.pop(g)

        return grouping

    def group_by_vegetation_ic(self, indices):
        """
        This function outputs a dictionary grouping environments with the same vegetation initial condition together
        """

        grouping = {}

        # Initialize groups with empty lists
        for t in self.vegetation_ics:
            grouping[t] = []

        # Group environments
        for i in indices:
            grouping[self.env_vegetation_ics[i]].append(i)

        # Remove any empty groups
        for g in list(grouping):
            if not grouping[g]:
                grouping.pop(g)

        return grouping
    
    def group_by_orientation(self, indices):
        """
        This function outputs a dictionary grouping environments with the same orientation together
        """

        grouping = {}

        # Initialize groups with empty lists
        for t in self.orientations:
            grouping[t] = []

        # Group environments
        for i in indices:
            grouping[self.env_orientation[i]].append(i)

        # Remove any empty groups
        for g in list(grouping):
            if not grouping[g]:
                grouping.pop(g)

        return grouping

    def reset(self, indices):
        """
        Reset given environments
        """


        # This function accepts both arrays and a single integer as input,
        # make sure we can process everything as an np array
        indices = np.array([indices]).flatten()

        # Set all hidden coefficients to zero
        self.hidden_states[indices, :, :, :] = 0

        # Reset all masks and conditions
        self.closed_mask_fullres[indices] = 0
        self.opened_mask_fullres[indices] = 0
        self.h_mask_fullres[indices] = 0
        self.h_cond_fullres[indices] = 0

        # Randomly choose a new type for each environment
        self.env_type[indices] = np.random.choice(self.types, indices.shape)
        self.env_water_strategy[indices] = np.random.choice(self.water_strategies, indices.shape)
        self.env_vegetation_ics[indices] = np.random.choice(self.vegetation_ics, indices.shape)
        self.env_orientation[indices] = np.random.choice(self.orientations, indices.shape)
        self.env_seed[indices] = 2.0 * math.pi * torch.floor(1000 * torch.rand(indices.shape))
        self.env_time[indices] = torch.zeros(indices.shape)

        # Helper function 1/4 -- Reset the type of environment (grouped)
        def reset_all_of_type(typename, group_indices):
            """
            group_indices is guaranteed to be non-empty
            """

            #
            # SALTMARSH SETTING
            #
            if typename == "numerical-saltmarsh":

                # Randomly select the right number of saltmarsh outputs
                group_size = len(group_indices)

                group_sample_idx = torch.randint(
                    low=0,
                    high=self.num_sfere_samples,
                    size=(group_size,)
                )

                # Collect the randomly selected SFERE outputs and move them to the GPU
                selected_sfere_outputs = self.sfere_snapshots[group_sample_idx].to(self.device)

                # Pull them through the Image-Fitting CNN and move the result back to CPU
                imfitted_sfere_outputs = self.imfit_net(selected_sfere_outputs).detach().cpu()

                #
                # Set the initial condition
                #
                self.hidden_states[group_indices] = imfitted_sfere_outputs

                # Remove all the water initially and set velocities to zero
                # self.hidden_states[group_indices, self.variables.get_slice_for("h")] = 0
                # self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")] = self.params.H0
                # self.hidden_states[group_indices, self.variables.get_slice_for("u")] = 0
                # self.hidden_states[group_indices, self.variables.get_slice_for("v")] = 0

                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1


            #
            # TOPOGRAPHIC FLAT
            #
            if typename == "topoflat":

                #
                # Initially, constant water level and zero constant sediment
                #

                # Reset the hidden state
                self.hidden_states[group_indices] = 0

                self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")] = self.params.H0

                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1


            #
            # TOPOGRAPHIC SLOPE
            #
            if typename == "toposlope":

                #
                # Initially, constant water level and slight slope in sedimentary topography
                #

                # Reset the hidden state
                self.hidden_states[group_indices] = 0

                self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")] = self.params.H0

                for x in range(self.width-1):
                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("s"),:,x] = (1 - (x/(self.width-1))) * 0.25


                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1

            #
            # TOPOGRAPHIC SLOPE, CURVED
            #
            if typename == "toposlope-curved":

                #
                # Initially, constant water level and slight slope in sedimentary topography
                #

                # Reset the hidden state
                self.hidden_states[group_indices] = 0

                self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")] = self.params.H0

                for x in range(self.width-1):
                    for y in range(self.height-1):

                        self.hidden_states[group_indices, self.variables.get_singular_slice_for("s"),y,x] = (1 - (x/(self.width-1)) + (0.005 * (y - self.height/2))**2) * 0.2


                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1

            #
            # TOPOGRAPHIC TRIANGLE WITH VEGETATION (SHARP)
            #
            if typename == "toposharp-vegmax":
                
                #
                # Initially, constant water level and slight slope in sedimentary topography
                #

                # Reset the hidden state
                self.hidden_states[group_indices] = 0

                self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")] = self.params.H0

                shift_distance = 20

                for x in range(self.width-1-shift_distance):
                    
                    # Upper triangle
                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("s"), :(self.height//2 - (x + shift_distance)//2), x] = 0.2
                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("b"), :(self.height//2 - (x + shift_distance)//2), x] = self.params.k

                    # Lower triangle
                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("s"), -(self.height//2 - (x + shift_distance)//2):, x] = 0.2
                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("b"), -(self.height//2 - (x + shift_distance)//2):, x] = self.params.k

                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1

            #
            # TOPOGRAPHIC TRIANGLE WITH VEGETATION SLOPE
            #
            if typename == "toposharp-vegslope":

                #
                # Initially, constant water level and slight slope in sedimentary topography
                #

                # Reset the hidden state
                self.hidden_states[group_indices] = 0

                self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")] = self.params.H0

                shift_distance = 10

                # Prepare vegetation slope
                sedmax = 0.2
                sedslope = torch.linspace(0, sedmax, self.height//2 - shift_distance//2).unsqueeze(0).unsqueeze(1).repeat(len(group_indices), 1, 1)
                vegslope = torch.linspace(0, self.params.k, self.height//2 - shift_distance//2).unsqueeze(0).unsqueeze(1).repeat(len(group_indices), 1, 1)

                for x in range(self.width-1-shift_distance):
                    
                    # Upper triangle
                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("s"), :(self.height//2 - (x + shift_distance)//2), x] = sedmax - sedslope[:, :, -(self.height//2 - (x + shift_distance)//2):]
                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("b"), :(self.height//2 - (x + shift_distance)//2), x] = self.params.k - vegslope[:, :, -(self.height//2 - (x + shift_distance)//2):]

                    # Lower triangle
                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("s"), -(self.height//2 - (x + shift_distance)//2):, x] = sedslope[:, :, :(self.height//2 - (x + shift_distance)//2)]
                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("b"), -(self.height//2 - (x + shift_distance)//2):, x] = vegslope[:, :, :(self.height//2 - (x + shift_distance)//2)]

                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1



        # Helper function 2/4 -- Reset the water strategy of the environment (grouped)
        def reset_all_of_water_strategy(water_strategy, group_indices):
            
            #
            # Hin water strategy
            #
            if water_strategy == "Hin":
                pass

            #
            # Tidal flow water strategy
            #
            if water_strategy == "tidal-flow":

                # Model tides at the open boundary
                self.h_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1
                self.h_cond_fullres[group_indices] = torch.clamp(self.params.wave_size * torch.sin(self.env_seed[group_indices]), min=0.01).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.width_fullres, self.height_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

        # Helper function 3/4 -- Reset the vegetation initial condition of the environment (grouped)
        def reset_all_of_vegetation_ic(vegetation_ic, group_indices):

            #
            # Random noise vegetation
            #
            if vegetation_ic == "uniform-noise":
                self.hidden_states[group_indices, self.variables.get_singular_slice_for("b")] = torch.rand_like(self.hidden_states[group_indices, self.variables.get_singular_slice_for("b")]) * self.params.k

            if vegetation_ic == "random-gaussians":
                # Randomly place some vegetation Gaussians
                xs = torch.arange(0, self.width-1)
                ys = torch.arange(0, self.height-1)
                x, y = torch.meshgrid(xs, ys, indexing='xy')

                gauss_stdev = 10

                for _ in range(5):
                    xpos = np.random.choice(range(self.width//8, 7*self.width//8+1), 1)[0]
                    ypos = np.random.choice(range(self.height//8, 7*self.height//8+1), 1)[0]

                    self.hidden_states[group_indices, self.variables.get_singular_slice_for("b")] = self.hidden_states[group_indices, self.variables.get_singular_slice_for("b")] \
                                                                                                    + self.params.k/2 * torch.exp(-(torch.pow(x - xpos, 2)/(2*gauss_stdev**2) + torch.pow(y - ypos, 2)/(2*gauss_stdev**2)))

            if vegetation_ic == "vd-vijsel":
                random_allocation = torch.rand_like(self.hidden_states[group_indices, self.variables.get_singular_slice_for("b")])
                vegetation_ic = torch.zeros_like(self.hidden_states[group_indices, self.variables.get_singular_slice_for("b")]).float()
                vegetation_ic[torch.where(random_allocation < self.params.pEst)] = self.params.k
                self.hidden_states[group_indices, self.variables.get_singular_slice_for("b")] = vegetation_ic

            if vegetation_ic == "empty":
                self.hidden_states[group_indices, self.variables.get_singular_slice_for("b")] = 0

            if vegetation_ic == "elsewhere-specified":
                pass


        # Helper function 4/4 -- Reset the orientation of environment (grouped)
        def rotate_all_of_orientation(orientation, group_indices):
            """
            group_indices is guaranteed to be non-empty
            """
            
            if orientation == "east":
                # East is the default orientation of every environment
                pass

            if orientation == "north":
                # Rotate PI/2 rads (1 turn) to orient the open boundary towards north
                # Rotate the initial condition
                self.hidden_states[group_indices] = torch.rot90(self.hidden_states[group_indices], k=1, dims=(2,3))

                # Swap u and v
                temp = self.hidden_states[group_indices, self.variables.get_slice_for("u")]
                self.hidden_states[group_indices, self.variables.get_slice_for("u")] = self.hidden_states[group_indices, self.variables.get_slice_for("v")]
                self.hidden_states[group_indices, self.variables.get_slice_for("v")] = temp

                # Multiply v by -1 to align the flow velocities with this new orientation
                self.hidden_states[group_indices, self.variables.get_slice_for("v")] *= -1

                # Rotate the boundary conditions
                self.closed_mask_fullres[group_indices] = torch.rot90(self.closed_mask_fullres[group_indices], k=1, dims=(2,3))
                self.opened_mask_fullres[group_indices] = torch.rot90(self.opened_mask_fullres[group_indices], k=1, dims=(2,3))
                self.h_mask_fullres[group_indices] = torch.rot90(self.h_mask_fullres[group_indices], k=1, dims=(2,3))
                self.h_cond_fullres[group_indices] = torch.rot90(self.h_cond_fullres[group_indices], k=1, dims=(2,3))

            if orientation == "west":
                # Rotate PI rads (2 turns) to orient the open boundary towards west
                # Rotate the initial condition
                self.hidden_states[group_indices] = torch.rot90(self.hidden_states[group_indices], k=2, dims=(2,3))

                # Multiply u and v by -1 to align the flow velocities with this new orientation
                self.hidden_states[group_indices, self.variables.get_slice_for("u")] *= -1
                self.hidden_states[group_indices, self.variables.get_slice_for("v")] *= -1

                # Rotate the boundary conditions
                self.closed_mask_fullres[group_indices] = torch.rot90(self.closed_mask_fullres[group_indices], k=2, dims=(2,3))
                self.opened_mask_fullres[group_indices] = torch.rot90(self.opened_mask_fullres[group_indices], k=2, dims=(2,3))
                self.h_mask_fullres[group_indices] = torch.rot90(self.h_mask_fullres[group_indices], k=2, dims=(2,3))
                self.h_cond_fullres[group_indices] = torch.rot90(self.h_cond_fullres[group_indices], k=2, dims=(2,3))
            
            if orientation == "south":
                # Rotate -PI/2 rads (-1 turns) to orient the open boundary towards south
                # Rotate the initial condition
                self.hidden_states[group_indices] = torch.rot90(self.hidden_states[group_indices], k=-1, dims=(2,3))

                # Swap u and v
                temp = self.hidden_states[group_indices, self.variables.get_slice_for("u")]
                self.hidden_states[group_indices, self.variables.get_slice_for("u")] = self.hidden_states[group_indices, self.variables.get_slice_for("v")]
                self.hidden_states[group_indices, self.variables.get_slice_for("v")] = temp

                # Multiply u by -1 to align the flow velocities with this new orientation
                self.hidden_states[group_indices, self.variables.get_slice_for("u")] *= -1

                # Rotate the boundary conditions
                self.closed_mask_fullres[group_indices] = torch.rot90(self.closed_mask_fullres[group_indices], k=-1, dims=(2,3))
                self.opened_mask_fullres[group_indices] = torch.rot90(self.opened_mask_fullres[group_indices], k=-1, dims=(2,3))
                self.h_mask_fullres[group_indices] = torch.rot90(self.h_mask_fullres[group_indices], k=-1, dims=(2,3))
                self.h_cond_fullres[group_indices] = torch.rot90(self.h_cond_fullres[group_indices], k=-1, dims=(2,3))
            
            

        # Group environments by their type [Groups are guaranteed to be non-empty]
        grouping = self.group_by_type(indices)
        for typename in grouping.keys():
            reset_all_of_type(typename, grouping[typename])

        # Group environments by their water strategy [Groups are guaranteed to be non-empty]
        grouping = self.group_by_water_strategy(indices)
        for typename in grouping.keys():
            reset_all_of_water_strategy(typename, grouping[typename])

        # Group environments by their vegetation initial condition type [Groups are guaranteed to be non-empty]
        grouping = self.group_by_vegetation_ic(indices)
        for typename in grouping.keys():
            reset_all_of_vegetation_ic(typename, grouping[typename])

        # Group environments by their orientation [Groups are guaranteed to be non-empty]
        grouping = self.group_by_orientation(indices)
        for orientation in grouping.keys():
            rotate_all_of_orientation(orientation, grouping[orientation])
    
        # Average pooling to create downsampled versions of the BCs
        self.closed_mask[indices] = F.avg_pool2d(self.closed_mask_fullres[indices],self.resolution_factor)
        self.opened_mask[indices] = F.avg_pool2d(self.opened_mask_fullres[indices],self.resolution_factor)
        self.h_mask[indices] = F.avg_pool2d(self.h_mask_fullres[indices],self.resolution_factor)
        self.h_cond[indices] = F.avg_pool2d(self.h_cond_fullres[indices],self.resolution_factor)



    def update(self, indices):
        """
        Update given environments
        """

        # This function accepts both arrays and a single integer as input,
        # make sure we can process everything as an np array
        indices = np.array([indices]).flatten()

        # Helper function 1/2 -- Update the type of environment (grouped)
        def update_all_of_type(typename, group_indices):
            """
            group_indices is guaranteed to be non-empty
            """
            
            pass

        # Helper function 2/2 -- Update the water strategy of environment (grouped)
        def update_all_of_water_strategy(water_strategy, group_indices):
            
            #
            # Hin water strategy
            #
            if water_strategy == "Hin":

                # Clamp the first order of h to minimum zero
                # This ensures the dry areas can be 'flooded' again by Hin as they would with correct tidal flow
                self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")] = torch.clamp(self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")], min=0)

                # Add Hin to the full domain
                self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")] = self.hidden_states[group_indices, self.variables.get_singular_slice_for("h")] + self.params.Hin

            #
            # Tidal flow water strategy
            #
            if water_strategy == "tidal-flow":

                self.h_cond_fullres[group_indices] = torch.clamp(self.params.wave_size * torch.sin(self.env_seed[group_indices] + self.env_time[group_indices]), min=0.01).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.width_fullres, self.height_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

        # Group environments by their type [Groups are guaranteed to be non-empty]
        grouping = self.group_by_type(indices)
        for typename in grouping.keys():
            update_all_of_type(typename, grouping[typename])

        # Group environments by their water strategy [Groups are guaranteed to be non-empty]
        grouping = self.group_by_water_strategy(indices)
        for typename in grouping.keys():
            update_all_of_water_strategy(typename, grouping[typename])
    
        # Average pooling to create downsampled versions of the BCs
        self.closed_mask[indices] = F.avg_pool2d(self.closed_mask_fullres[indices],self.resolution_factor)
        self.opened_mask[indices] = F.avg_pool2d(self.opened_mask_fullres[indices],self.resolution_factor)
        self.h_mask[indices] = F.avg_pool2d(self.h_mask_fullres[indices],self.resolution_factor)
        self.h_cond[indices] = F.avg_pool2d(self.h_cond_fullres[indices],self.resolution_factor)
        
        # Update the time for each environment
        self.env_time[indices] = self.env_time[indices] + math.pi / 10.0
        

    def ask(self):
        """
		:return:
			grids:
				hidden_state					-> shape: bs x hidden_size x (w-1) x (h-1)
				boundary-features:
					u_cond						-> shape: bs x 1 x w x h
					u_mask (continuous) 		-> shape: bs x 1 x w x h differentiable renderer would allow for differentiable geometries
					v_cond						-> shape: bs x 1 x w x h
					v_mask (continuous) 		-> shape: bs x 1 x w x h differentiable renderer would allow for differentiable geometries
			sample-grids:
				- grid-offsets (x,y,t) 			-> shape: bs x 3 x 1 x 1 (values between 0,1; all offsets are the same within an "image" - otherwise: bsx3xwxh)
				- sample_u_cond					-> shape: bs x 1 x w x h
				- sample_u_mask (boolean)		-> shape: bs x 1 x w x h
				- sample_v_cond					-> shape: bs x 1 x w x h
				- sample_v_mask (boolean)		-> shape: bs x 1 x w x h
		"""

        # Store which indices we gather in the batch, so we can
        # update the corresponding environments upon 'tell' after 'ask'
        self.asked_indices = np.random.choice(self.dataset_size, self.batch_size)

        # Update the environments before sending them out
        self.update(self.asked_indices)

        # Compute grid offsets and sample BCs
        grid_offsets = []
        sample_closed_mask = []
        sample_opened_mask = []
        sample_h_mask = []
        sample_h_cond = []

        for _ in range(self.n_samples):

            # Grid offsets
            offset = torch.rand(3)
            grid_offsets.append(offset)

            x_offset = min(int(self.resolution_factor*offset[0]),self.resolution_factor-1)
            y_offset = min(int(self.resolution_factor*offset[1]),self.resolution_factor-1)

            sample_closed_mask.append(self.closed_mask_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_opened_mask.append(self.opened_mask_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_h_mask.append(self.h_mask_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_h_cond.append(self.h_cond_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])

        # Move all data to the desired device
        for i in range(self.n_samples):
            grid_offsets[i] = grid_offsets[i].to(self.device)
            sample_closed_mask[i] = sample_closed_mask[i].to(self.device)
            sample_opened_mask[i] = sample_opened_mask[i].to(self.device)
            sample_h_mask[i] = sample_h_mask[i].to(self.device)
            sample_h_cond[i] = sample_h_cond[i].to(self.device)

        # Return the hidden states and boundary conditions after moving them to the desired device
        return self.hidden_states[self.asked_indices].to(self.device), \
                self.closed_mask[self.asked_indices].to(self.device), \
                self.opened_mask[self.asked_indices].to(self.device), \
                self.h_mask[self.asked_indices].to(self.device), \
                self.h_cond[self.asked_indices].to(self.device), \
                grid_offsets, \
                sample_closed_mask, \
                sample_opened_mask, \
                sample_h_mask, \
                sample_h_cond
    
    def tell(self, hidden_states):

        # Update hidden states after moving them back to the CPU
        self.hidden_states[self.asked_indices] = hidden_states.detach().cpu()

        # Randomly reset environments
        self.t += 1
		#print(f"t: {self.t} - {(self.average_sequence_length/self.batch_size)}")
        if self.t % int(self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
            self.reset(int(self.i))
            self.i = (self.i+1)%self.dataset_size

        # Warming up: We reset the entire batch with increasing interval at the start of training
        # self.warmup_t += 1
        
        # if self.warmup_t == self.warmup_reset_at:
        #     self.reset(self.asked_indices)
        #     self.warmup_t = 0
        #     self.warmup_reset_at *= 2 # We reset the entire batch every 1, 2, 4, 8, 16, 32, ..., 128, ..., 1024 iterations






    #
    # Data related tasks
    #

    def interpolate_states(self, old_hidden_states, new_hidden_states, offset):
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
        old_h, old_grad_h, _ = self.variables["h"].interpolate_at(self.variables.extract_from(old_hidden_states, "h"), offset[:2])
        new_h, new_grad_h, _ = self.variables["h"].interpolate_at(self.variables.extract_from(new_hidden_states, "h"), offset[:2])

        # u field: requires first derivative + laplace
        old_u, old_grad_u, old_laplacian_u = self.variables["u"].interpolate_at(self.variables.extract_from(old_hidden_states, "u"), offset[:2])
        new_u, new_grad_u, new_laplacian_u = self.variables["u"].interpolate_at(self.variables.extract_from(new_hidden_states, "u"), offset[:2])

        # v field: requires first derivative + laplace
        old_v, old_grad_v, old_laplacian_v = self.variables["v"].interpolate_at(self.variables.extract_from(old_hidden_states, "v"), offset[:2])
        new_v, new_grad_v, new_laplacian_v = self.variables["v"].interpolate_at(self.variables.extract_from(new_hidden_states, "v"), offset[:2])

        # s field: requires first derivative
        old_s, old_grad_s, old_laplacian_s = self.variables["s"].interpolate_at(self.variables.extract_from(old_hidden_states, "s"), offset[:2])
        new_s, new_grad_s, new_laplacian_s = self.variables["s"].interpolate_at(self.variables.extract_from(new_hidden_states, "s"), offset[:2])

        # b field: requires first derivative
        old_b, old_grad_b, old_laplacian_b = self.variables["b"].interpolate_at(self.variables.extract_from(old_hidden_states, "b"), offset[:2])
        new_b, new_grad_b, new_laplacian_b = self.variables["b"].interpolate_at(self.variables.extract_from(new_hidden_states, "b"), offset[:2])

        # First order interpolation in time
        h = (1-offset[2])*old_h + offset[2]*new_h
        grad_h = (1-offset[2])*old_grad_h + offset[2]*new_grad_h
        dh_dt = (new_h - old_h) / self.params.dt

        u = (1-offset[2])*old_u + offset[2]*new_u
        grad_u = (1-offset[2])*old_grad_u + offset[2]*new_grad_u
        laplacian_u = (1-offset[2])*old_laplacian_u + offset[2]*new_laplacian_u
        du_dt = (new_u - old_u) / self.params.dt

        v = (1-offset[2])*old_v + offset[2]*new_v
        grad_v = (1-offset[2])*old_grad_v + offset[2]*new_grad_v
        laplacian_v = (1-offset[2])*old_laplacian_v + offset[2]*new_laplacian_v
        dv_dt = (new_v - old_v) / self.params.dt

        s = (1-offset[2])*old_s + offset[2]*new_s
        grad_s = (1-offset[2])*old_grad_s + offset[2]*new_grad_s
        laplacian_s = (1-offset[2])*old_laplacian_s + offset[2]*new_laplacian_s
        ds_dt = (new_s - old_s) / self.params.dt

        b = (1-offset[2])*old_b + offset[2]*new_b
        grad_b = (1-offset[2])*old_grad_b + offset[2]*new_grad_b
        laplacian_b = (1-offset[2])*old_laplacian_b + offset[2]*new_laplacian_b
        db_dt = (new_b - old_b) / self.params.dt
        
        return h, grad_h, dh_dt, u, grad_u, laplacian_u, du_dt, v, grad_v, laplacian_v, dv_dt, s, grad_s, laplacian_s, ds_dt, b, grad_b, laplacian_b, db_dt
    

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

        # b field: requires first derivative + laplace
        b, grad_b, _ = self.variables["b"].interpolate_superres_at(self.variables.extract_from(hidden_states, "b"), resolution_factor)

        return h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b