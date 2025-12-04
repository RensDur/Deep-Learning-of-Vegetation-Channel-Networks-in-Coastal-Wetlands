from statistics import correlation

import torch
import numpy as np
import math



class Dataset:
    def __init__(self, params, device):

        # Dimensions
        self.width = params.width
        self.height = params.height
        self.dx = params.separation
        self.dy = params.separation

        self.batch_size = params.batch_size
        self.dataset_size = params.dataset_size

        #
        # Store local copy of the parameters
        #
        self.params = params

        # Fields (scalar fields)
        self.h = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.u = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.v = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.S = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.B = torch.zeros(self.dataset_size, 1, self.height, self.width)

        # Domain boundaries
        # ==> Future addition

        # Time tracking per environment
        self.t = torch.zeros(self.dataset_size)

        # Reset all environments upon initialization
        self.reset(range(self.dataset_size))

        # Preselect pytorch device
        self.device = device


    def reset(self, indices):
        """
        METHOD: Reset environments
        """

        # Uniform water thickness H0
        self.h[indices, 0, :, :] = self.params.H0

        # Flow velocities are zero everywhere
        self.u[indices, 0, :, :] = 0
        self.v[indices, 0, :, :] = 0

        # Sedimentary elevation is zero everywhere
        self.S[indices, 0, :, :] = 0
        
        # Vegetation density is zero everywhere, except for some randomly placed tussocks
        mask = torch.zeros_like(self.B, dtype=torch.bool)
        mask[indices, 0] = torch.rand(len(indices), self.height, self.width) < self.params.pEst
        self.B[mask] = self.params.k

        # Reset time-tracking for each selected environment
        self.t[indices] = 0

    def update(self, indices):
        """
        Update selected environments
        """

        # Enforce boundary condition effects

        # Minimum water layer thickness
        self.h[indices, :, :, :] = torch.clamp(self.h[indices, :, :, :], min=self.params.Hc)

        # Left boundary
        self.u[indices, :, :, 0] = -self.u[indices, :, :, 1]
        self.v[indices, :, :, 0] = self.v[indices, :, :, 1]
        self.h[indices, :, :, 0] = self.h[indices, :, :, 1]
        self.S[indices, :, :, 0] = self.S[indices, :, :, 1]
        self.B[indices, :, :, 0] = self.B[indices, :, :, 1]

        # Right boundary: closed
        # self.u[indices, :, :, -1] = -self.u[indices, :, :, -2]
        # self.v[indices, :, :, -1] = self.v[indices, :, :, -2]
        # self.h[indices, :, :, -1] = self.h[indices, :, :, -2]
        # self.S[indices, :, :, -1] = self.S[indices, :, :, -2]
        # self.B[indices, :, :, -1] = self.B[indices, :, :, -2]

        # Right boundary: open
        self.u[indices, :, :, -1] = 2*self.u[indices, :, :, -2] - self.u[indices, :, :, -3]
        self.v[indices, :, :, -1] = 2*self.v[indices, :, :, -2] - self.v[indices, :, :, -3]
        self.h[indices, :, :, -1] = self.h[indices, :, :, -2]
        self.S[indices, :, :, -1] = self.S[indices, :, :, -2]
        self.B[indices, :, :, -1] = self.B[indices, :, :, -2]

        # Top
        self.u[indices, :, 0, :] = self.u[indices, :, 1, :]
        self.v[indices, :, 0, :] = -self.v[indices, :, 1, :]
        self.h[indices, :, 0, :] = self.h[indices, :, 1, :]
        self.S[indices, :, 0, :] = self.S[indices, :, 1, :]
        self.B[indices, :, 0, :] = self.B[indices, :, 1, :]

        # Bottom
        self.u[indices, :, -1, :] = self.u[indices, :, -2, :]
        self.v[indices, :, -1, :] = -self.v[indices, :, -2, :]
        self.h[indices, :, -1, :] = self.h[indices, :, -2, :]
        self.S[indices, :, -1, :] = self.S[indices, :, -2, :]
        self.B[indices, :, -1, :] = self.B[indices, :, -2, :]


    #
    # ASK & TELL
    #
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

        # Return the chosen batch
        return  self.h[self.asked_indices].to(self.device), \
                self.u[self.asked_indices].to(self.device), \
                self.v[self.asked_indices].to(self.device), \
                self.S[self.asked_indices].to(self.device), \
                self.B[self.asked_indices].to(self.device)

    def tell(self, h, u, v, S, B, random_reset=False):
        """
        Return the updated state to the dataset
        :param h: updated h
        :param u: updated u
        :param v: updated v
        :param S: updated S
        :param B: updated B
        """

        # Update state
        self.h[self.asked_indices] = h.detach().cpu()
        self.u[self.asked_indices] = u.detach().cpu()
        self.v[self.asked_indices] = v.detach().cpu()
        self.S[self.asked_indices] = S.detach().cpu()
        self.B[self.asked_indices] = B.detach().cpu()

        # Update time-tracking
        self.t[self.asked_indices] += self.params.dt

        if random_reset:
            # Reset an environment randomly after progressing a certain amount of time
            r = np.random.uniform(0, 1, self.batch_size)

            reset_indices = [self.asked_indices[i] for i in range(self.batch_size) if r[i] < 0.01]
            self.reset(reset_indices)

