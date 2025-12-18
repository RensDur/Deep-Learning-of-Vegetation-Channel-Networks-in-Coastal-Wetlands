import torch
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
        self.cond_mask = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.h_cond = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.cond_mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.h_cond_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)

        # Environment information
        self.types = [
            "rest-lake",
            "oscillator"
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

        # BC: Standard frame
        self.cond_mask_fullres[indices] = 1
        self.cond_mask_fullres[indices, :, self.padding_fullres:-self.padding_fullres, self.padding_fullres:-self.padding_fullres] = 0

        # For each environment, randomly choose one of the types
        for index in indices:
            t = np.random.choice(self.types)
            self.env_info[index]["type"] = t


    def update(self, indices):
        """
        Update given environments
        """
        pass
        

    def ask(self):
        """
        Ask for a batch of boundary- and initial-conditions
        :return: batch
        """

        # Store which indices we gather in the batch, so we can
        # update the corresponding environments upon 'tell' after 'ask'
        self.asked_indices = np.random.choice(self.dataset_size, self.batch_size)

        # Update the environments before sending them out
        self.update(self.asked_indices)

        # Return the hidden states and boundary conditions
        return self.hidden_states[self.asked_indices]
    
    def tell(self, hidden_states):
        self.hidden_states[self.asked_indices] = hidden_states.detach()