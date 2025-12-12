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
            SplineVariable("u", 2, requires_derivative=True, device=self.device),
            SplineVariable("v", 2, requires_derivative=True, device=self.device),
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
        self.cond_mask = torch.zeros(self.dataset_size, 1, self.width, self.height).int()
        self.h_cond = 

    def reset(self, indices):
        """
        Reset given environments
        """

        # Set all hidden coefficients to zero
        self.hidden_states[indices, :, :, :] = 0

    def update(self, indices):
        """
        Update given environments
        """

        # This method is not yet needed...
        pass

    def ask(self) -> LatentBatch:
        """
        Ask for a batch of boundary- and initial-conditions
        :return: batch
        """

        # Store which indices we gather in the batch, so we can
        # update the corresponding environments upon 'tell' after 'ask'
        self.asked_indices = np.random.choice(self.dataset_size, self.batch_size)

        # Update the environments before sending them out
        self.update(self.asked_indices)

        # Create a LatentBatch from the selected environments
        return LatentBatch(self.hidden_states[self.asked_indices])
    
    def tell(self, batch: LatentBatch):

        self.hidden_states[self.asked_indices] = batch.hidden_states.detach()