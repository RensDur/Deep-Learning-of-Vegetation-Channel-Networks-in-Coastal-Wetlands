import torch
import torch.nn.functional as F
import numpy as np
from spline.spline_variable import SplineVariable



class Dataset:
    
    def __init__(self, params, device=torch.device("cpu")):

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
        self.variables = [
            SplineVariable("h", 2, requires_derivative=True, device=self.device),
            SplineVariable("hu", 2, requires_derivative=True, device=self.device),
            SplineVariable("hv", 2, requires_derivative=True, device=self.device),
        ]

        # Compute the total hidden size
        self.hidden_size = np.sum([svar.hidden_size() for svar in self.variables])

        # Hidden state
        self.hidden_states = torch.zeros(
            self.dataset_size,
            self.hidden_size,
            self.width-1,
            self.height-1
        )

        # Boundary conditions and masking
        self.u_mask = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.v_mask = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.u_cond = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.v_cond = torch.zeros(self.dataset_size, 1, self.width, self.height)

        self.u_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.v_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.u_cond_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.v_cond_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)

        # Environment information
        self.types = [
            "rest-lake",
            # "single-oscillator",
            # "multiple-oscillators"
        ]
        self.env_info = [{} for _ in range(self.dataset_size)]

        # Environment resetting
        self.t = 0
        self.i = 0

        # Reset all environments
        self.reset(range(self.dataset_size))

    def reset(self, indices):
        """
        Reset given environments
        """

        # Set all hidden coefficients to zero
        self.hidden_states[indices, :, :, :] = 0

        # BC: Standard frame in terms of u and v
        self.u_mask_fullres[indices] = 0 # BCs on u only apply on the left- and rightmost strip of padding
        self.u_mask_fullres[indices,:,:,:self.padding_fullres] = 1
        self.u_mask_fullres[indices,:,:,-self.padding_fullres:] = 1

        self.v_mask_fullres[indices] = 0 # BCs on v only apply on the top- and bottom-most strip of padding
        self.v_mask_fullres[indices,:,:self.padding_fullres,:] = 1
        self.v_mask_fullres[indices,:,-self.padding_fullres:,:] = 1

        # For each environment, randomly choose one of the types
        for index in indices:
            self.env_info[index]["type"] = np.random.choice(self.types)

            #
            # LAKE AT REST
            #
            if self.env_info[index]["type"] == "rest-lake":

                # In a lake at rest, with closed boundaries, no flow velocities at the boundaries they apply to
                self.u_cond_fullres[indices] = 0
                self.v_cond_fullres[indices] = 0

                self.u_cond_fullres[indices] *= self.u_mask_fullres[indices]
                self.v_cond_fullres[indices] *= self.v_mask_fullres[indices]

    
            # Average pooling to create downsampled versions of the BCs
            self.u_cond[index:(index+1)] = F.avg_pool2d(self.u_cond_full_res[index:(index+1)],self.resolution_factor)
            self.u_mask[index:(index+1)] = F.avg_pool2d(self.u_mask_full_res[index:(index+1)],self.resolution_factor)
            self.v_cond[index:(index+1)] = F.avg_pool2d(self.v_cond_full_res[index:(index+1)],self.resolution_factor)
            self.v_mask[index:(index+1)] = F.avg_pool2d(self.v_mask_full_res[index:(index+1)],self.resolution_factor)



    def update(self, indices):
        """
        Update given environments
        """
        pass
        

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
        sample_u_cond = []
        sample_u_mask = []
        sample_v_cond = []
        sample_v_mask = []

        for _ in range(self.n_samples):

            # Grid offsets
            offset = torch.rand(3)
            grid_offsets.append(offset)

            x_offset = min(int(self.resolution_factor*offset[0]),self.resolution_factor-1) # TODO: Isn't this just the same as floor(self.resolution_factor*offset[0])?
            y_offset = min(int(self.resolution_factor*offset[1]),self.resolution_factor-1)

            sample_u_cond.append(self.u_cond_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_u_mask.append(self.u_mask_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_v_cond.append(self.v_cond_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
            sample_v_mask.append(self.v_mask_fullres[self.asked_indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])

        # Return the hidden states and boundary conditions
        return self.hidden_states[self.asked_indices], \
                self.u_cond[self.asked_indices], \
                self.u_mask[self.asked_indices], \
                self.v_cond[self.asked_indices], \
                self.v_mask[self.asked_indices], \
                grid_offsets, \
                sample_u_cond, \
                sample_u_mask, \
                sample_v_cond, \
                sample_v_mask,
    
    def tell(self, hidden_states):

        # Update hidden states
        self.hidden_states[self.asked_indices] = hidden_states.detach()

        # Randomly reset environments
        self.t += 1
		#print(f"t: {self.t} - {(self.average_sequence_length/self.batch_size)}")
        if self.t % int(self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
            self.reset(int(self.i))
            self.i = (self.i+1)%self.dataset_size