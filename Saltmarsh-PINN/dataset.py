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

        self.padding = 5

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
            SplineVariable("h", 1),                           # h describes the zero-meaned surface height, on top of H0
            SplineVariable("u", 2),
            SplineVariable("v", 2),
            device=self.device
        )

        # Hidden state
        self.hidden_states = torch.zeros(
            self.dataset_size,
            self.variables.hidden_size(),
            self.height+1,
            self.width+1,
        )

        # Boundary conditions and masking
        self.h_mask = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.h_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.uv_mask = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.uv_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)

        # Loss image per environment in the dataset
        # This acts as a Probability Density Function (PDF)
        # In reset(), the values are set to 1/width*height
        self.loss_tensors = torch.zeros(self.dataset_size, self.height, self.width)

        # Environment information
        self.types = [
            "rest-lake",
            "oscillator",
            "random-oscillator",
            "multiple-random-oscillator",
            "reflection",
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
        self.h_mask[indices] = 0
        self.h_cond[indices] = 0

        # BC: h holds around the entire frame
        self.uv_mask[indices] = 1
        self.uv_mask[indices, :, self.padding:-self.padding, self.padding:-self.padding] = 0

        # Velocity condition zero
        self.uv_cond[indices] = 0

        # Randomly choose a new type for each environment
        self.env_type[indices] = np.random.choice(self.types, indices.shape)
        self.env_seed[indices] = 2.0 * math.pi * torch.floor(1000 * torch.rand(indices.shape))
        self.env_time[indices] = torch.zeros(indices.shape)

        # Reset loss images
        self.loss_tensors[indices] = 1.0 / (self.width * self.height) # Loss tensors should represent a PDF

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
                        self.h_mask[group_indices,:,(self.height//2+x-5):(self.height//2+x+5),(self.width//2+y-5):(self.width//2+y+5)] = 1
                        self.uv_mask[group_indices,:,(self.height//2+x-5):(self.height//2+x+5),(self.width//2+y-5):(self.width//2+y+5)] = 1

                # Set the masks and conditions
                self.h_cond[group_indices,:,self.padding:-self.padding, self.padding:-self.padding] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height - 2*self.padding, self.width - 2*self.padding)
                self.h_cond[group_indices] = self.h_cond[group_indices] * self.h_mask[group_indices]

            #
            # RANDOMLY PLACED OSCILLATOR
            #
            if typename == "random-oscillator":
                # obstabcles (oscillators)
                for x in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in np.random.choice(range(-45, 46), 1):#[-45,-15,15,45]:
                        self.h_mask[group_indices,:,(self.height//2+x-5):(self.height//2+x+5),(self.width//2+y-5):(self.width//2+y+5)] = 1
                        self.uv_mask[group_indices,:,(self.height//2+x-5):(self.height//2+x+5),(self.width//2+y-5):(self.width//2+y+5)] = 1

                # Set the masks and conditions
                self.h_cond[group_indices,:,self.padding:-self.padding, self.padding:-self.padding] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height - 2*self.padding, self.width - 2*self.padding)
                self.h_cond[group_indices] = self.h_cond[group_indices] * self.h_mask[group_indices]

            #
            # RANDOMLY PLACED OSCILLATOR
            #
            if typename == "multiple-random-oscillator":
                # obstabcles (oscillators)
                for x in np.random.choice(range(-45, 46, 5), 2):#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in np.random.choice(range(-45, 46, 5), 2):#[-45,-15,15,45]:
                        self.h_mask[group_indices,:,(self.height//2+x-5):(self.height//2+x+5),(self.width//2+y-5):(self.width//2+y+5)] = 1
                        self.uv_mask[group_indices,:,(self.height//2+x-5):(self.height//2+x+5),(self.width//2+y-5):(self.width//2+y+5)] = 1

                # Set the masks and conditions
                self.h_cond[group_indices,:,self.padding:-self.padding, self.padding:-self.padding] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height - 2*self.padding, self.width - 2*self.padding)
                self.h_cond[group_indices] = self.h_cond[group_indices] * self.h_mask[group_indices]

            #
            # REFLECTION
            #
            if typename == "reflection":

                # obstabcles (oscillators)
                for x in [-10]:#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
                    for y in [60]:#[-45,-15,15,45]:
                        self.h_mask[group_indices,:,(self.height//2+x-5):(self.height//2+x+5),(self.width//2+y-5):(self.width//2+y+5)] = 1
                        self.uv_mask[group_indices,:,(self.height//2+x-5):(self.height//2+x+5),(self.width//2+y-5):(self.width//2+y+5)] = 1

                # Set the masks and conditions
                self.h_cond[group_indices,:,self.padding:-self.padding, self.padding:-self.padding] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height - 2*self.padding, self.width - 2*self.padding)
                self.h_cond[group_indices] = self.h_cond[group_indices] * self.h_mask[group_indices]

                # We install a barrier starting in the top-center going towards the middle of the domain of thickness 10
                barrier_thickness = 10
                self.uv_mask[group_indices,:, 0:(self.height//2), (self.width//2-barrier_thickness//2):(self.width//2+barrier_thickness//2)+1] = 1

                # Set the masks and conditions
                self.uv_cond[group_indices,:,self.padding:-self.padding, self.padding:-self.padding] = 0
                self.uv_cond[group_indices] = self.uv_cond[group_indices] * self.h_mask[group_indices]

        for typename in grouping.keys():
            reset_all_of_type(typename, grouping[typename])



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
            if typename == "oscillator" or typename == "random-oscillator" or typename == "multiple-random-oscillator" or typename == "reflection":
                self.h_cond[group_indices,:,self.padding:-self.padding,self.padding:-self.padding] = self.params.wave_size * torch.sin(self.env_seed[group_indices] + self.env_time[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height - 2*self.padding, self.width - 2*self.padding)
                self.h_cond[group_indices] = self.h_cond[group_indices] * self.h_mask[group_indices]

        for typename in grouping.keys():
            reset_all_of_type(typename, grouping[typename])
        
        # Update the time for each environment
        self.env_time[indices] = self.env_time[indices] + math.pi / 10.0

    def sample_conditions(self, indices, offsets, strat="nearest-neighbour"):
        """
        :indices: selected batch indices from the dataset
        :offsets: offsets at which we would like to draw a sample. Shape (batch_size x 2:{x,y} x num_samples)
        :return: condition sample for each variable in this dataset. Shape (batch_size x num_channels x num_samples)
                 where the num_channels refers to the number of conditioned variables or derivations thereof
        """

        # This function accepts both arrays and a single integer as input,
        # make sure we can process everything as an np array
        indices = np.array([indices]).flatten()

        batch_size = indices.shape[0]
        num_samples = offsets.shape[2]

        # Quick check to validate the number of selected indices matches the offset size
        assert batch_size == offsets.shape[0]

        # 1. Apply Nearest Neighbour to find integer offsets within [width, height]
        offsets_nn = torch.round(offsets - 0.5).clamp(min=0).long() # Subtract 1/2 because we sample in the center of each cell

        # 2. Gather the selected environments' BCs and masks
        sample_h_cond = self.h_cond[indices]
        sample_h_mask = self.h_mask[indices]
        sample_uv_cond = self.uv_cond[indices]
        sample_uv_mask = self.uv_mask[indices]

        # 3. Extract the samples from these environments
        batch_indices = torch.arange(batch_size)[:, None].expand(batch_size, num_samples)

        sample_h_cond = sample_h_cond[batch_indices, :, offsets_nn[:, 1], offsets_nn[:, 0]].swapdims(1,2)
        sample_h_mask = sample_h_mask[batch_indices, :, offsets_nn[:, 1], offsets_nn[:, 0]].swapdims(1,2)
        sample_uv_cond = sample_uv_cond[batch_indices, :, offsets_nn[:, 1], offsets_nn[:, 0]].swapdims(1,2)
        sample_uv_mask = sample_uv_mask[batch_indices, :, offsets_nn[:, 1], offsets_nn[:, 0]].swapdims(1,2)
            
        # 4. Return the samples
        return sample_h_cond, \
            sample_h_mask, \
            sample_uv_cond, \
            sample_uv_mask

        

    def ask(self):
        """
		:return:
			grids:
				hidden_state					-> shape: bs x hidden_size x (h+1) x (w+1)
				boundary-features:
					u_cond						-> shape: bs x 1 x h x w
					u_mask               		-> shape: bs x 1 x h x w
					v_cond						-> shape: bs x 1 x h x w
					v_mask               		-> shape: bs x 1 x h x w
			sample-grids:
				- sample_offsets (x,y,t) 		-> shape: bs x num_samples x 3:{x,y,t}
				- sample_u_cond					-> shape: bs x num_samples x 1
				- sample_u_mask (boolean)		-> shape: bs x num_samples x 1
				- sample_v_cond					-> shape: bs x num_samples x 1
				- sample_v_mask (boolean)		-> shape: bs x num_samples x 1
		"""

        # Store which indices we gather in the batch, so we can
        # update the corresponding environments upon 'tell' after 'ask'
        self.asked_indices = np.random.choice(self.dataset_size, self.batch_size)

        # Update the environments before sending them out
        self.update(self.asked_indices)

        # Generate random sample offsets
        sample_offsets = torch.rand(self.batch_size, 3, self.n_samples)
        
        # Scale sample offsets in x and y by the size of the domain
        sample_offsets[:, 0:2, :] = sample_offsets[:, 0:2, :] * self.width # TODO: Assume width == height!

        # Sample at these offsets to obtain boundary conditions
        sample_h_cond, sample_h_mask, sample_uv_cond, sample_uv_mask = self.sample_conditions(self.asked_indices, sample_offsets)

        # Return the hidden states and boundary conditions after moving them to the desired device
        return self.hidden_states[self.asked_indices].to(self.device), \
                self.h_cond[self.asked_indices].to(self.device), \
                self.h_mask[self.asked_indices].to(self.device), \
                self.uv_cond[self.asked_indices].to(self.device), \
                self.uv_mask[self.asked_indices].to(self.device), \
                sample_offsets.to(self.device), \
                sample_h_cond.to(self.device), \
                sample_h_mask.to(self.device), \
                sample_uv_cond.to(self.device), \
                sample_uv_mask.to(self.device)
    
    def tell(self, hidden_states, loss_tensors):
        """
        :loss_tensors: shape (batch_size, height, width)
        """

        # Update hidden states after moving them back to the CPU.
        # Cast to storage dtype so AMP (float16/bfloat16) outputs don't break in-place assign.
        self.hidden_states[self.asked_indices] = hidden_states.detach().cpu().float()

        # Update the loss images per environment in the batch
        self.loss_tensors[self.asked_indices] = (loss_tensors / torch.sum(loss_tensors, dim=(1,2))).detach().cpu().float()

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

    def interpolate_states_at_regular_interval(self, old_hidden_states, new_hidden_states, offset):
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
        old_h_group = self.variables["h"].interpolate_at_regular_interval(self.variables.extract_from(old_hidden_states, "h"), offset[:2], include_derivative=True)
        new_h_group = self.variables["h"].interpolate_at_regular_interval(self.variables.extract_from(new_hidden_states, "h"), offset[:2], include_derivative=True)

        old_h, old_grad_h = old_h_group[:, 0:1], old_h_group[:, 1:3]
        new_h, new_grad_h = new_h_group[:, 0:1], new_h_group[:, 1:3]

        # u field: requires first derivative + laplace
        old_u_group = self.variables["u"].interpolate_at_regular_interval(self.variables.extract_from(old_hidden_states, "u"), offset[:2], include_derivative=True, include_laplacian=True)
        new_u_group = self.variables["u"].interpolate_at_regular_interval(self.variables.extract_from(new_hidden_states, "u"), offset[:2], include_derivative=True, include_laplacian=True)

        old_u, old_grad_u, old_laplace_u = old_u_group[:, 0:1], old_u_group[:, 1:3], old_u_group[:, 3:4]
        new_u, new_grad_u, new_laplace_u = new_u_group[:, 0:1], new_u_group[:, 1:3], new_u_group[:, 3:4]

        # v field: requires first derivative + laplace
        old_v_group = self.variables["v"].interpolate_at_regular_interval(self.variables.extract_from(old_hidden_states, "v"), offset[:2], include_derivative=True, include_laplacian=True)
        new_v_group = self.variables["v"].interpolate_at_regular_interval(self.variables.extract_from(new_hidden_states, "v"), offset[:2], include_derivative=True, include_laplacian=True)

        old_v, old_grad_v, old_laplace_v = old_v_group[:, 0:1], old_v_group[:, 1:3], old_v_group[:, 3:4]
        new_v, new_grad_v, new_laplace_v = new_v_group[:, 0:1], new_v_group[:, 1:3], new_v_group[:, 3:4]

        # First order interpolation in time
        h = (1-offset[2])*old_h + offset[2]*new_h
        grad_h = (1-offset[2])*old_grad_h + offset[2]*new_grad_h
        dh_dt = (new_h - old_h) / self.params.dt

        u = (1-offset[2])*old_u + offset[2]*new_u
        grad_u = (1-offset[2])*old_grad_u + offset[2]*new_grad_u
        laplace_u = (1-offset[2])*old_laplace_u + offset[2]*new_laplace_u
        du_dt = (new_u - old_u) / self.params.dt

        v = (1-offset[2])*old_v + offset[2]*new_v
        grad_v = (1-offset[2])*old_grad_v + offset[2]*new_grad_v
        laplace_v = (1-offset[2])*old_laplace_v + offset[2]*new_laplace_v
        dv_dt = (new_v - old_v) / self.params.dt
        
        return h, grad_h, dh_dt, u, grad_u, laplace_u, du_dt, v, grad_v, laplace_v, dv_dt

    def interpolate_states_at(self, old_hidden_states, new_hidden_states, offsets):
        """
        :old_hidden_states: old hidden states (size: bs x (v_size+p_size) x w x h)
        :new_hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
        :offset: offset in x / y / t direction (size: bs x num_samples x 3:<x,y,t>)
        :return: interpolated fields for all variables in this dataset
        """

        # z field: requires first derivative
        old_h_group = self.variables["h"].interpolate_at(self.variables.extract_from(old_hidden_states, "h"), offsets[:,:2], include_derivative=True)
        new_h_group = self.variables["h"].interpolate_at(self.variables.extract_from(new_hidden_states, "h"), offsets[:,:2], include_derivative=True)

        old_h, old_grad_h = old_h_group[:, 0:1], old_h_group[:, 1:3]
        new_h, new_grad_h = new_h_group[:, 0:1], new_h_group[:, 1:3]

        # u field: requires first derivative + laplace
        old_u_group = self.variables["u"].interpolate_at(self.variables.extract_from(old_hidden_states, "u"), offsets[:,:2], include_derivative=True, include_laplacian=True)
        new_u_group = self.variables["u"].interpolate_at(self.variables.extract_from(new_hidden_states, "u"), offsets[:,:2], include_derivative=True, include_laplacian=True)

        old_u, old_grad_u, old_laplace_u = old_u_group[:, 0:1], old_u_group[:, 1:3], old_u_group[:, 3:4]
        new_u, new_grad_u, new_laplace_u = new_u_group[:, 0:1], new_u_group[:, 1:3], new_u_group[:, 3:4]

        # v field: requires first derivative + laplace
        old_v_group = self.variables["v"].interpolate_at(self.variables.extract_from(old_hidden_states, "v"), offsets[:,:2], include_derivative=True, include_laplacian=True)
        new_v_group = self.variables["v"].interpolate_at(self.variables.extract_from(new_hidden_states, "v"), offsets[:,:2], include_derivative=True, include_laplacian=True)

        old_v, old_grad_v, old_laplace_v = old_v_group[:, 0:1], old_v_group[:, 1:3], old_v_group[:, 3:4]
        new_v, new_grad_v, new_laplace_v = new_v_group[:, 0:1], new_v_group[:, 1:3], new_v_group[:, 3:4]

        # All above interpolation-results have shape <batch_size x num_samples x num_channels>
        # where the number of channels is 1 for a scalar field and 2 for a vector field (i.e. gradients)
        #
        # Offsets have shape <batch_size x num_samples x 3:{x,y,t}>

        # First order interpolation in time
        h = (1-offsets[:,2:3])*old_h + offsets[:,2:3]*new_h
        grad_h = (1-offsets[:,2:3])*old_grad_h + offsets[:,2:3]*new_grad_h
        dh_dt = (new_h - old_h) / self.params.dt

        u = (1-offsets[:,2:3])*old_u + offsets[:,2:3]*new_u
        grad_u = (1-offsets[:,2:3])*old_grad_u + offsets[:,2:3]*new_grad_u
        laplace_u = (1-offsets[:,2:3])*old_laplace_u + offsets[:,2:3]*new_laplace_u
        du_dt = (new_u - old_u) / self.params.dt

        v = (1-offsets[:,2:3])*old_v + offsets[:,2:3]*new_v
        grad_v = (1-offsets[:,2:3])*old_grad_v + offsets[:,2:3]*new_grad_v
        laplace_v = (1-offsets[:,2:3])*old_laplace_v + offsets[:,2:3]*new_laplace_v
        dv_dt = (new_v - old_v) / self.params.dt

        # Resulting fields now have shape <batch_size x num_samples x num_channels>
        return h, grad_h, dh_dt, u, grad_u, laplace_u, du_dt, v, grad_v, laplace_v, dv_dt
    

    def interpolate_states_highres(self, hidden_states, width, height):
        """
        :hidden_states: new hidden states (size: bs x (v_size+p_size) x H x W)
        :width: output image width
        :height: output image height
        :return: interpolated fields for all variables in this dataset
        """

        # h field: requires first derivative
        h = self.variables["h"].interpolate_highres(self.variables.extract_from(hidden_states, "h"), width, height)

        # u field: requires first derivative + laplace
        u = self.variables["u"].interpolate_highres(self.variables.extract_from(hidden_states, "u"), width, height)

        # v field: requires first derivative + laplace
        v = self.variables["v"].interpolate_highres(self.variables.extract_from(hidden_states, "v"), width, height)

        return h, u, v