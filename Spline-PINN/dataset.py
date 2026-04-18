import torch
import torch.nn.functional as F
import numpy as np
import math
from spline.spline_variable import SplineVariable
from spline.spline_array import SplineArray


class Dataset:
    
    def __init__(self, params, device=torch.device("cpu"), types=None, orientations=None):

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
            SplineVariable("u", 2, requires_derivative=True, requires_laplacian=True),
            SplineVariable("v", 2, requires_derivative=True, requires_laplacian=True),
            SplineVariable("s", 2, requires_derivative=True, requires_laplacian=True),
            SplineVariable("b", 2, requires_derivative=True, requires_laplacian=True),
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

        self.closed_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.opened_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)

        # Load the saltmarsh numerical solution, pre-fitted to a hidden spline representation
        self.prefit_saltmarsh_250_000 = torch.load(f"numerical_spline_converted/{self.variables.summary()}/250000/hidden_state.pt").cpu()
        self.prefit_saltmarsh_300_000 = torch.load(f"numerical_spline_converted/{self.variables.summary()}/300000/hidden_state.pt").cpu()
        self.prefit_saltmarsh_650_000 = torch.load(f"numerical_spline_converted/{self.variables.summary()}/650000/hidden_state.pt").cpu()
        self.prefit_saltmarsh_1_300_000 = torch.load(f"numerical_spline_converted/{self.variables.summary()}/1300000/hidden_state.pt").cpu()
        self.prefit_saltmarsh_2_500_000 = torch.load(f"numerical_spline_converted/{self.variables.summary()}/2500000/hidden_state.pt").cpu()

        # Environment information
        self.types = [
            "numerical-saltmarsh-250_000",
            "numerical-saltmarsh-300_000",
            "numerical-saltmarsh-650_000",
            "numerical-saltmarsh-1_300_000",
            "numerical-saltmarsh-2_500_000",
        ] if types is None else types

        self.orientations = [
            "north",
            "east",
            "south",
            "west"
        ] if orientations is None else orientations

        print(f"Running with types {self.types} and orientations {self.orientations}")

        self.env_type = np.random.choice(self.types, self.dataset_size)
        self.env_orientation = np.random.choice(self.orientations, self.dataset_size)
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

        # Randomly choose a new type for each environment
        self.env_type[indices] = np.random.choice(self.types, indices.shape)
        self.env_orientation[indices] = np.random.choice(self.orientations, indices.shape)
        self.env_seed[indices] = 2.0 * math.pi * torch.floor(1000 * torch.rand(indices.shape))
        self.env_time[indices] = torch.zeros(indices.shape)

        # Helper function 1/2 -- Reset the type of environment (grouped)
        def reset_all_of_type(typename, group_indices):
            """
            group_indices is guaranteed to be non-empty
            """

            #
            # SALTMARSH SETTING
            #
            if typename == "numerical-saltmarsh-250_000":

                #
                # Set the initial condition
                #
                self.hidden_states[group_indices] = self.prefit_saltmarsh_250_000.clone()

                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1

            
            if typename == "numerical-saltmarsh-300_000":

                #
                # Set the initial condition
                #
                self.hidden_states[group_indices] = self.prefit_saltmarsh_300_000.clone()

                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1

            if typename == "numerical-saltmarsh-650_000":

                #
                # Set the initial condition
                #
                self.hidden_states[group_indices] = self.prefit_saltmarsh_650_000.clone()

                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1

            if typename == "numerical-saltmarsh-1_300_000":

                #
                # Set the initial condition
                #
                self.hidden_states[group_indices] = self.prefit_saltmarsh_1_300_000.clone()

                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1

            if typename == "numerical-saltmarsh-2_500_000":

                #
                # Set the initial condition
                #
                self.hidden_states[group_indices] = self.prefit_saltmarsh_2_500_000.clone()

                #
                # Set the boundary conditions
                #

                # All sides are closed, except the right edge
                self.closed_mask_fullres[group_indices] = 1
                self.closed_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,self.padding_fullres:] = 0

                # The right edge is open
                self.opened_mask_fullres[group_indices,:,self.padding_fullres:-self.padding_fullres,-self.padding_fullres:] = 1

        # Helper function 2/2 -- Reset the orientation of environment (grouped)
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
            
            

        # Group environments by their type [Groups are guaranteed to be non-empty]
        grouping = self.group_by_type(indices)
        for typename in grouping.keys():
            reset_all_of_type(typename, grouping[typename])

        # Group environments by their orientation [Groups are guaranteed to be non-empty]
        grouping = self.group_by_orientation(indices)
        for orientation in grouping.keys():
            rotate_all_of_orientation(orientation, grouping[orientation])
    
        # Average pooling to create downsampled versions of the BCs
        self.closed_mask[indices] = F.avg_pool2d(self.closed_mask_fullres[indices],self.resolution_factor)
        self.opened_mask[indices] = F.avg_pool2d(self.opened_mask_fullres[indices],self.resolution_factor)



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
            pass

        for typename in grouping.keys():
            reset_all_of_type(typename, grouping[typename])
    
        # Average pooling to create downsampled versions of the BCs
        self.closed_mask[indices] = F.avg_pool2d(self.closed_mask_fullres[indices],self.resolution_factor)
        self.opened_mask[indices] = F.avg_pool2d(self.opened_mask_fullres[indices],self.resolution_factor)
        
        # Update the time for each environment
        self.env_time[indices] = self.env_time[indices] + math.pi / 100.0
        

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

        for _ in range(self.n_samples):

            # Grid offsets
            offset = torch.rand(3)
            grid_offsets.append(offset)

            x_offset = min(int(self.resolution_factor*offset[0]),self.resolution_factor-1)
            y_offset = min(int(self.resolution_factor*offset[1]),self.resolution_factor-1)

            sample_closed_mask.append(self.closed_mask_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_opened_mask.append(self.opened_mask_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])

        # Move all data to the desired device
        for i in range(self.n_samples):
            grid_offsets[i] = grid_offsets[i].to(self.device)
            sample_closed_mask[i] = sample_closed_mask[i].to(self.device)
            sample_opened_mask[i] = sample_opened_mask[i].to(self.device)

        # Return the hidden states and boundary conditions after moving them to the desired device
        return self.hidden_states[self.asked_indices].to(self.device), \
                self.closed_mask[self.asked_indices].to(self.device), \
                self.opened_mask[self.asked_indices].to(self.device), \
                grid_offsets, \
                sample_closed_mask, \
                sample_opened_mask
    
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