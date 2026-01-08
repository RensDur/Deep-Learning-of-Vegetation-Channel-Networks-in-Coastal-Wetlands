import torch
import torch.nn.functional as F
import numpy as np
from spline.spline_variable import SplineVariable
from spline.spline_array import SplineArray


class Dataset:
    
    def __init__(self, params, device=torch.device("cpu")):

        # Local copy of the parameters
        self.params = params

        # Dimensions
        self.width = params.width
        self.height = params.height
        self.resolution_factor = params.resolution_factor
        self.width_fullres = self.resolution_factor * self.width
        self.height_fullres = self.resolution_factor * self.height

        self.padding = 4
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
        self.h_mask = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.h_cond = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.uv_mask = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.uv_cond = torch.zeros(self.dataset_size, 1, self.width, self.height)

        self.h_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.h_cond_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.uv_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.uv_cond_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)

        # Environment information
        self.types = [
            "rest-lake",
            "oscillator",
            # "multiple-oscillators"
        ]

        self.env_type = np.random.choice(self.types, self.dataset_size)
        self.env_seed = 1000 * torch.rand(self.dataset_size)
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

        # BC: h holds around the entire frame
        self.h_mask_fullres[indices] = 0 # Initially we don't impose conditions on h anywhere in the domain
        
        self.uv_mask_fullres[indices] = 1
        self.uv_mask_fullres[indices, :, self.padding:-self.padding, self.padding:-self.padding] = 0

        # Velocity condition zero
        self.uv_cond_fullres[indices] = 0

        # Randomly choose a new type for each environment
        self.env_type[indices] = np.random.choice(self.types, indices.shape)
        self.env_seed[indices] = 1000 * torch.rand(indices.shape)
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
                        self.h_mask_fullres[group_indices,:,(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor),(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor)] = 1
                        self.uv_mask_fullres[group_indices,:,(self.width_fullres//2+(-5+x)*self.resolution_factor):(self.width_fullres//2+(5+x)*self.resolution_factor),(self.height_fullres//2+(-5+y)*self.resolution_factor):(self.height_fullres//2+(5+y)*self.resolution_factor)] = 1

                # Set the masks and conditions
                self.h_cond_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres, self.padding_fullres:-self.padding_fullres] = self.params.wave_size * torch.sin(self.env_seed[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.width_fullres - 2*self.padding_fullres, self.height_fullres - 2*self.padding_fullres)
                self.h_cond_fullres[group_indices] *= self.h_mask_fullres[group_indices]

        for typename in grouping.keys():
            reset_all_of_type(typename, grouping[typename])

        # Soften the transition planes
        # Create sponge BCs by applying a gradient in the boundary
        conv_kernel = torch.tensor([[0, 0.25, 0],
                                    [0.25, 0, 0.25],
                                    [0, 0.25, 0]], device=self.device).view(1, 1, 3, 3)
        
        for _ in range(2):
            self.h_mask_fullres[indices] = 1-F.conv2d(1-self.h_mask_fullres[indices], conv_kernel, padding=1)
            self.uv_mask_fullres[indices] = 1-F.conv2d(1-self.uv_mask_fullres[indices], conv_kernel, padding=1)
    
        # Average pooling to create downsampled versions of the BCs
        self.h_cond[indices] = F.avg_pool2d(self.h_cond_fullres[indices],self.resolution_factor)
        self.h_mask[indices] = F.avg_pool2d(self.h_mask_fullres[indices],self.resolution_factor)
        self.uv_cond[indices] = F.avg_pool2d(self.uv_cond_fullres[indices],self.resolution_factor)
        self.uv_mask[indices] = F.avg_pool2d(self.uv_mask_fullres[indices],self.resolution_factor)



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
            if typename == "oscillator":
                self.h_cond_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:-self.padding_fullres] = self.params.wave_size * torch.sin(self.env_seed[group_indices] + self.env_time[group_indices]).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.width_fullres - 2*self.padding_fullres, self.height_fullres - 2*self.padding_fullres)
                self.h_cond_fullres[group_indices] *= self.h_mask_fullres[group_indices]

        for typename in grouping.keys():
            reset_all_of_type(typename, grouping[typename])
    
        # Average pooling to create downsampled versions of the BCs
        self.h_cond[indices] = F.avg_pool2d(self.h_cond_fullres[indices],self.resolution_factor)
        self.h_mask[indices] = F.avg_pool2d(self.h_mask_fullres[indices],self.resolution_factor)
        self.uv_cond[indices] = F.avg_pool2d(self.uv_cond_fullres[indices],self.resolution_factor)
        self.uv_mask[indices] = F.avg_pool2d(self.uv_mask_fullres[indices],self.resolution_factor)
        
        # Update the time for each environment
        self.env_time[indices] += 0.01
        

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
        sample_h_cond = []
        sample_h_mask = []
        sample_uv_cond = []
        sample_uv_mask = []

        for _ in range(self.n_samples):

            # Grid offsets
            offset = torch.rand(3)
            grid_offsets.append(offset)

            x_offset = min(int(self.resolution_factor*offset[0]),self.resolution_factor-1) # TODO: Isn't this just the same as floor(self.resolution_factor*offset[0])?
            y_offset = min(int(self.resolution_factor*offset[1]),self.resolution_factor-1)

            sample_h_cond.append(self.h_cond_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_h_mask.append(self.h_mask_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_uv_cond.append(self.uv_cond_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_uv_mask.append(self.uv_mask_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])

        # Move all data to the desired device
        for i in range(self.n_samples):
            grid_offsets[i] = grid_offsets[i].to(self.device)
            sample_h_cond[i] = sample_h_cond[i].to(self.device)
            sample_h_mask[i] = sample_h_mask[i].to(self.device)
            sample_uv_cond[i] = sample_uv_cond[i].to(self.device)
            sample_uv_mask[i] = sample_uv_mask[i].to(self.device)

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
                sample_uv_mask
    
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
        self.warmup_t += 1
        
        if self.warmup_t == self.warmup_reset_at:
            self.reset(self.asked_indices)
            self.warmup_reset_at *= 2 # We reset the entire batch every 1, 2, 4, 8, 16, 32, ..., 128, ..., 1024 iterations






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
        old_u, old_grad_u, old_laplace_u = self.variables["u"].interpolate_at(self.variables.extract_from(old_hidden_states, "u"), offset[:2])
        new_u, new_grad_u, new_laplace_u = self.variables["u"].interpolate_at(self.variables.extract_from(new_hidden_states, "u"), offset[:2])

        # v field: requires first derivative + laplace
        old_v, old_grad_v, old_laplace_v = self.variables["v"].interpolate_at(self.variables.extract_from(old_hidden_states, "v"), offset[:2])
        new_v, new_grad_v, new_laplace_v = self.variables["v"].interpolate_at(self.variables.extract_from(new_hidden_states, "v"), offset[:2])

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
        u, grad_u, laplace_u = self.variables["u"].interpolate_superres_at(self.variables.extract_from(hidden_states, "u"), resolution_factor)

        # v field: requires first derivative + laplace
        v, grad_v, laplace_v = self.variables["v"].interpolate_superres_at(self.variables.extract_from(hidden_states, "v"), resolution_factor)

        return h, grad_h, u, grad_u, laplace_u, v, grad_v, laplace_v