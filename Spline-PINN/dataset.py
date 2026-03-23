import torch
import torch.nn.functional as F
import numpy as np
import math
from spline.spline_variable import SplineVariable
from spline.spline_array import SplineArray


class Dataset:
    
    def __init__(self, params, device=torch.device("cpu"), types=None):

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
            SplineVariable("hu", 2, requires_derivative=True),
            SplineVariable("hv", 2, requires_derivative=True),
            # SplineVariable("s", 1, requires_derivative=True, requires_laplacian=True),
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
        self.h_mask = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.h_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.uv_mask = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.uv_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.s_mask = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.s_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)

        self.h_mask_fullres = torch.zeros(self.dataset_size, 1, self.height_fullres, self.width_fullres)
        self.h_cond_fullres = torch.zeros(self.dataset_size, 1, self.height_fullres, self.width_fullres)
        self.uv_mask_fullres = torch.zeros(self.dataset_size, 1, self.height_fullres, self.width_fullres)
        self.uv_cond_fullres = torch.zeros(self.dataset_size, 1, self.height_fullres, self.width_fullres)
        self.s_mask_fullres = torch.zeros(self.dataset_size, 1, self.height_fullres, self.width_fullres)
        self.s_cond_fullres = torch.zeros(self.dataset_size, 1, self.height_fullres, self.width_fullres)

        # Water inflow per environment
        self.h_in = torch.zeros(self.dataset_size, 1, self.height, self.width)

        # Environment information
        self.types = [
            "rest-lake",
            "oscillator",
            "random-oscillator",
            "multiple-random-oscillator",
            "four-corners-oscillator",
            "reflection",
            "top-edge-oscillator",
            "bottom-edge-oscillator",
            "left-edge-oscillator",
            "right-edge-oscillator",
            # "top-open-outflow",
            # "bottom-open-outflow",
            # "right-open-outflow",
            # "left-open-outflow",
            # "top-open-outflow-obstacle",
            # "bottom-open-outflow-obstacle",
            # "right-open-outflow-obstacle",
            # "left-open-outflow-obstacle",
            # "multiple-oscillators"
        ] if types is None else types

        print(f"Running with types: {self.types}")

        self.env_type = np.random.choice(self.types, self.dataset_size)
        self.env_seed = 2.0 * math.pi * torch.floor(1000 * torch.rand(self.dataset_size))
        self.env_time = torch.zeros(self.dataset_size)

        # Environment resetting
        self.t = 0
        self.i = 0
        self.warmup_t = 0
        self.warmup_reset_at = 1

        # Reset all environments
        print("Resetting all environments")
        self.reset(range(self.dataset_size))

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
        self.h_mask_fullres[indices] = 0
        self.h_cond_fullres[indices] = 0

        # BC: h holds around the entire frame
        self.uv_mask_fullres[indices] = 1
        self.uv_mask_fullres[indices, :, self.padding_fullres:-self.padding_fullres, self.padding_fullres:-self.padding_fullres] = 0

        # Velocity condition zero
        self.uv_cond_fullres[indices] = 0

        # S condition zero
        self.s_mask_fullres[indices] = 0
        self.s_cond_fullres[indices] = 0

        # Randomly choose a new type for each environment
        self.env_type[indices] = np.random.choice(self.types, indices.shape)
        self.env_seed[indices] = 2.0 * math.pi * torch.floor(1000 * torch.rand(indices.shape))
        self.env_time[indices] = torch.zeros(indices.shape)

        # Group environments by their type [Groups are guaranteed to be non-empty]
        grouping = self.group_by_type(indices)

        # Helper function
        def reset_all_of_type(typename, group_indices):
            """
            group_indices is guaranteed to be non-empty
            """

            #
            # LAKE AT REST
            #
            if typename == "rest-lake":
                pass

            #
            # OSCILLATOR
            #
            if typename == "oscillator":

                # obstabcles (oscillators)
                for x in [0]:#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in [0]:#[-45,-15,15,45]:
                        self.h_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1
                        self.uv_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres, self.padding_fullres:-self.padding_fullres] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres - 2*self.padding_fullres, self.width_fullres - 2*self.padding_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            #
            # RANDOMLY PLACED OSCILLATOR
            #
            if typename == "random-oscillator":
                # obstabcles (oscillators)
                for x in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:
                        self.h_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1
                        self.uv_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres, self.padding_fullres:-self.padding_fullres] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres - 2*self.padding_fullres, self.width_fullres - 2*self.padding_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            #
            # RANDOMLY PLACED OSCILLATOR
            #
            if typename == "multiple-random-oscillator":
                # obstabcles (oscillators)
                for x in np.random.choice(range(-45, 46, 5), 2):#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in np.random.choice(range(-45, 46, 5), 2):#[-45,-15,15,45]:
                        self.h_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1
                        self.uv_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres, self.padding_fullres:-self.padding_fullres] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres - 2*self.padding_fullres, self.width_fullres - 2*self.padding_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            #
            # EVENLY SPACED FOUR CORNERS
            #
            if typename == "four-corners-oscillator":
                # obstabcles (oscillators)
                for x in [-50, 50]:
                    for y in [-50, 50]:
                        self.h_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1
                        self.uv_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres, self.padding_fullres:-self.padding_fullres] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres - 2*self.padding_fullres, self.width_fullres - 2*self.padding_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            #
            # REFLECTION
            #
            if typename == "reflection":

                # obstabcles (oscillators)
                for x in [-10]:#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in [60]:#[-45,-15,15,45]:
                        self.h_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1
                        self.uv_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres, self.padding_fullres:-self.padding_fullres] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres - 2*self.padding_fullres, self.width_fullres - 2*self.padding_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

                # We install a barrier starting in the top-center going towards the middle of the domain of thickness 10
                barrier_thickness = 10 * self.resolution_factor
                self.uv_mask_fullres[group_indices,:, 0:(self.height_fullres//2), (self.width_fullres//2-barrier_thickness//2):(self.width_fullres//2+barrier_thickness//2)+1] = 1

                # Set the masks and conditions
                self.uv_cond_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres, self.padding_fullres:-self.padding_fullres] = 0
                self.uv_cond_fullres[group_indices] = self.uv_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            #
            # EDGE OSCILLATORS
            #
            if typename == "top-edge-oscillator":
                self.h_mask_fullres[group_indices,:,:self.padding_fullres,self.padding_fullres:-self.padding_fullres] = 1
                self.uv_mask_fullres[group_indices,:,:self.padding_fullres,self.padding_fullres:-self.padding_fullres] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,:,:] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres, self.width_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            if typename == "bottom-edge-oscillator":
                self.h_mask_fullres[group_indices,:,-self.padding_fullres:,self.padding_fullres:-self.padding_fullres] = 1
                self.uv_mask_fullres[group_indices,:,-self.padding_fullres:,self.padding_fullres:-self.padding_fullres] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,:,:] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres, self.width_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            if typename == "left-edge-oscillator":
                self.h_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,:self.padding_fullres] = 1
                self.uv_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,:self.padding_fullres] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,:,:] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres, self.width_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            if typename == "right-edge-oscillator":
                self.h_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1
                self.uv_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,:,:] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres, self.width_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            #
            # OPEN OUTFLOW BOUNDARIES
            #
            if typename == "top-open-outflow":
                # Water flowing into the environment
                self.h_in[group_indices,:,:,:] = self.params.Hin

                # Open boundary: Remove the closed boundary and add S=0
                self.uv_mask_fullres[group_indices,:,:self.padding_fullres,:] = 0
                self.s_mask_fullres[group_indices,:,:self.padding_fullres,:] = 1

            if typename == "bottom-open-outflow":
                # Water flowing into the environment
                self.h_in[group_indices,:,:,:] = self.params.Hin

                # Open boundary: Remove the closed boundary and add S=0
                self.uv_mask_fullres[group_indices,:,-self.padding_fullres:,:] = 0
                self.s_mask_fullres[group_indices,:,-self.padding_fullres:,:] = 1

            if typename == "right-open-outflow":
                # Water flowing into the environment
                self.h_in[group_indices,:,:,:] = self.params.Hin

                # Open boundary: Remove the closed boundary and add S=0
                self.uv_mask_fullres[group_indices,:,:,-self.padding_fullres:] = 0
                self.s_mask_fullres[group_indices,:,:,-self.padding_fullres:] = 1

            if typename == "left-open-outflow":
                # Water flowing into the environment
                self.h_in[group_indices,:,:,:] = self.params.Hin

                # Open boundary: Remove the closed boundary and add S=0
                self.uv_mask_fullres[group_indices,:,:,:self.padding_fullres] = 0
                self.s_mask_fullres[group_indices,:,:,:self.padding_fullres] = 1

            #
            # OPEN OUTFLOW BOUNDARIES WITH A RANDOMLY PLACED OBSTACLE
            #
            if typename == "top-open-outflow-obstacle":
                # Water flowing into the environment
                self.h_in[group_indices,:,:,:] = self.params.Hin

                # Open boundary: Remove the closed boundary and add S=0
                self.uv_mask_fullres[group_indices,:,:self.padding_fullres,:] = 0
                self.s_mask_fullres[group_indices,:,:self.padding_fullres,:] = 1

                # obstabcle (pillars)
                for x in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:
                        self.uv_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1

            if typename == "bottom-open-outflow-obstacle":
                # Water flowing into the environment
                self.h_in[group_indices,:,:,:] = self.params.Hin

                # Open boundary: Remove the closed boundary and add S=0
                self.uv_mask_fullres[group_indices,:,-self.padding_fullres:,:] = 0
                self.s_mask_fullres[group_indices,:,-self.padding_fullres:,:] = 1

                # obstabcle (pillars)
                for x in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:
                        self.uv_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1

            if typename == "right-open-outflow-obstacle":
                # Water flowing into the environment
                self.h_in[group_indices,:,:,:] = self.params.Hin

                # Open boundary: Remove the closed boundary and add S=0
                self.uv_mask_fullres[group_indices,:,:,-self.padding_fullres:] = 0
                self.s_mask_fullres[group_indices,:,:,-self.padding_fullres:] = 1

                # obstabcle (pillars)
                for x in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:
                        self.uv_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1

            if typename == "left-open-outflow-obstacle":
                # Water flowing into the environment
                self.h_in[group_indices,:,:,:] = self.params.Hin

                # Open boundary: Remove the closed boundary and add S=0
                self.uv_mask_fullres[group_indices,:,:,:self.padding_fullres] = 0
                self.s_mask_fullres[group_indices,:,:,:self.padding_fullres] = 1

                # obstabcle (pillars)
                for x in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:
                        self.uv_mask_fullres[group_indices,:,(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor),(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor)] = 1


        for typename in grouping.keys():
            reset_all_of_type(typename, grouping[typename])

        # Soften the transition planes
        # Create sponge BCs by applying a gradient in the boundary
        # conv_kernel = torch.tensor([[0, 0.25, 0],
        #                             [0.25, 0, 0.25],
        #                             [0, 0.25, 0]]).view(1, 1, 3, 3)
        
        # for _ in range(2):
        #     self.h_mask_fullres[indices] = 1-F.conv2d(1-self.h_mask_fullres[indices], conv_kernel, padding=1)
        #     self.uv_mask_fullres[indices] = 1-F.conv2d(1-self.uv_mask_fullres[indices], conv_kernel, padding=1)
    
        # Average pooling to create downsampled versions of the BCs
        self.h_cond[indices] = F.avg_pool2d(self.h_cond_fullres[indices],self.resolution_factor)
        self.h_mask[indices] = F.avg_pool2d(self.h_mask_fullres[indices],self.resolution_factor)
        self.uv_cond[indices] = F.avg_pool2d(self.uv_cond_fullres[indices],self.resolution_factor)
        self.uv_mask[indices] = F.avg_pool2d(self.uv_mask_fullres[indices],self.resolution_factor)
        self.s_cond[indices] = F.avg_pool2d(self.s_cond_fullres[indices],self.resolution_factor)
        self.s_mask[indices] = F.avg_pool2d(self.s_mask_fullres[indices],self.resolution_factor)



    def update(self, indices):
        """
        Update given environments
        """

        # This function accepts both arrays and a single integer as input,
        # make sure we can process everything as an np array
        indices = np.array([indices]).flatten()

        # Group environments by their type [Groups are guaranteed to be non-empty]
        grouping = self.group_by_type(indices)

        # Helper function
        def reset_all_of_type(typename, group_indices):
            """
            group_indices is guaranteed to be non-empty
            """

            #
            # OSCILLATOR
            #
            if typename == "oscillator" or typename == "random-oscillator" or typename == "multiple-random-oscillator" or typename == "four-corners-oscillator" or typename == "reflection":
                self.h_cond_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:-self.padding_fullres] = self.params.wave_size * torch.sin(self.env_seed[group_indices] + self.env_time[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres - 2*self.padding_fullres, self.width_fullres - 2*self.padding_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

            if typename == "top-edge-oscillator" or typename == "bottom-edge-oscillator" or typename == "left-edge-oscillator" or typename == "right-edge-oscillator":
                self.h_cond_fullres[group_indices,:,:,:] = self.params.wave_size * torch.sin(self.env_seed[group_indices] + self.env_time[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height_fullres, self.width_fullres)
                self.h_cond_fullres[group_indices] = self.h_cond_fullres[group_indices] * self.h_mask_fullres[group_indices]

        for typename in grouping.keys():
            reset_all_of_type(typename, grouping[typename])
    
        # Average pooling to create downsampled versions of the BCs
        self.h_cond[indices] = F.avg_pool2d(self.h_cond_fullres[indices],self.resolution_factor)
        self.h_mask[indices] = F.avg_pool2d(self.h_mask_fullres[indices],self.resolution_factor)
        self.uv_cond[indices] = F.avg_pool2d(self.uv_cond_fullres[indices],self.resolution_factor)
        self.uv_mask[indices] = F.avg_pool2d(self.uv_mask_fullres[indices],self.resolution_factor)
        self.s_cond[indices] = F.avg_pool2d(self.s_cond_fullres[indices],self.resolution_factor)
        self.s_mask[indices] = F.avg_pool2d(self.s_mask_fullres[indices],self.resolution_factor)
        
        # Update the time for each environment
        self.env_time[indices] = self.env_time[indices] + math.pi / 7.0
        

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
				- grid-offsets (t, y, x) 		-> shape: bs x 3 x 1 x 1 (values between 0,1; all offsets are the same within an "image" - otherwise: bsx3xwxh)
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
        sample_h_cond = []
        sample_h_mask = []
        sample_uv_cond = []
        sample_uv_mask = []
        sample_s_cond = []
        sample_s_mask = []

        for _ in range(self.n_samples):

            # Grid offsets
            offset = torch.rand(3)
            grid_offsets.append(offset)

            y_offset = min(int(self.resolution_factor*offset[1]),self.resolution_factor-1)
            x_offset = min(int(self.resolution_factor*offset[2]),self.resolution_factor-1)

            sample_h_cond.append(self.h_cond_fullres[self.asked_indices,:,y_offset::self.resolution_factor,x_offset::self.resolution_factor])
            sample_h_mask.append(self.h_mask_fullres[self.asked_indices,:,y_offset::self.resolution_factor,x_offset::self.resolution_factor])
            sample_uv_cond.append(self.uv_cond_fullres[self.asked_indices,:,y_offset::self.resolution_factor,x_offset::self.resolution_factor])
            sample_uv_mask.append(self.uv_mask_fullres[self.asked_indices,:,y_offset::self.resolution_factor,x_offset::self.resolution_factor])
            sample_s_cond.append(self.s_cond_fullres[self.asked_indices,:,y_offset::self.resolution_factor,x_offset::self.resolution_factor])
            sample_s_mask.append(self.s_mask_fullres[self.asked_indices,:,y_offset::self.resolution_factor,x_offset::self.resolution_factor])

        # Move all data to the desired device
        for i in range(self.n_samples):
            grid_offsets[i] = grid_offsets[i].to(self.device)
            sample_h_cond[i] = sample_h_cond[i].to(self.device)
            sample_h_mask[i] = sample_h_mask[i].to(self.device)
            sample_uv_cond[i] = sample_uv_cond[i].to(self.device)
            sample_uv_mask[i] = sample_uv_mask[i].to(self.device)
            sample_s_cond[i] = sample_s_cond[i].to(self.device)
            sample_s_mask[i] = sample_s_mask[i].to(self.device)

        # Return the hidden states and boundary conditions after moving them to the desired device
        return self.hidden_states[self.asked_indices].to(self.device), \
                self.h_cond[self.asked_indices].to(self.device), \
                self.h_mask[self.asked_indices].to(self.device), \
                self.uv_cond[self.asked_indices].to(self.device), \
                self.uv_mask[self.asked_indices].to(self.device), \
                grid_offsets, \
                sample_h_cond, \
                sample_h_mask, \
                sample_uv_cond, \
                sample_uv_mask,
    
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
        :offset: offset in (t, y, x) direction (vector of size 3 containing values between 0 and 1)
        :return: interpolated fields for:
            :z: z field
            :grad(z): gradient of z field
            :laplace(z): laplacian of z field
            :dz/dt: velocity of z field
            :dz^2/dt^2: acceleration of z field
        """

        # h field: requires first derivative
        h, grad_h, _ = self.variables["h"].interpolate_at(
            self.variables.extract_from(old_hidden_states, "h"),
            self.variables.extract_from(new_hidden_states, "h"),
            offset
        )

        # hu field: requires first derivative
        hu, grad_hu, _ = self.variables["hu"].interpolate_at(
            self.variables.extract_from(old_hidden_states, "hu"),
            self.variables.extract_from(new_hidden_states, "hu"),
            offset
        )

        # hv field: requires first derivative
        hv, grad_hv, _ = self.variables["hv"].interpolate_at(
            self.variables.extract_from(old_hidden_states, "hv"),
            self.variables.extract_from(new_hidden_states, "hv"),
            offset
        )

        # s field: requires first derivative
        # s, grad_s, laplacian_s = self.variables["s"].interpolate_at(
        #     self.variables.extract_from(old_hidden_states, "s"),
        #     self.variables.extract_from(new_hidden_states, "s"),
        #     offset
        # )

        #
        # Extract the time derivative and spatial derivatives
        #
        dh_dt = grad_h[:, 0:1]
        grad_h = grad_h[:, 1:3]

        dhu_dt = grad_hu[:, 0:1]
        grad_hu = grad_hu[:, 1:3]

        dhv_dt = grad_hv[:, 0:1]
        grad_hv = grad_hv[:, 1:3]

        # ds_dt = grad_s[:, 0:1]
        # grad_s = grad_s[:, 1:3]
        
        return h, grad_h, dh_dt, hu, grad_hu, dhu_dt, hv, grad_hv, dhv_dt
    

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
        hu, grad_hu, _ = self.variables["hu"].interpolate_superres_at(self.variables.extract_from(hidden_states, "hu"), resolution_factor)

        # v field: requires first derivative + laplace
        hv, grad_hv, _ = self.variables["hv"].interpolate_superres_at(self.variables.extract_from(hidden_states, "hv"), resolution_factor)

        # s field: requires first derivative + laplace
        # s, grad_s, _ = self.variables["s"].interpolate_superres_at(self.variables.extract_from(hidden_states, "s"), resolution_factor)

        return h, grad_h, hu, grad_hu, hv, grad_hv