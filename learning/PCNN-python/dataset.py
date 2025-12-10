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
        # self.S = torch.zeros(self.dataset_size, 1, self.height, self.width)
        # self.B = torch.zeros(self.dataset_size, 1, self.height, self.width)

        # Domain boundaries
        self.cond_mask = torch.zeros(self.dataset_size, 1, self.height, self.width) # Boundary mask - [1 - Omega] - 1 on the boundaries, 0 everywhere else
        self.h_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.u_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.v_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)

        # Time tracking per environment
        self.t = torch.zeros(self.dataset_size)

        self.average_sequence_length = params.average_sequence_length
        self.average_sequence_t = 0
        self.average_sequence_i = 0

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

        x = torch.linspace(0, self.width * self.dx, self.width)
        y = torch.linspace(0, self.height * self.dy, self.height)
        x, y = torch.meshgrid(x, y, indexing='xy')

        for _ in range(3):
            x_mu = np.random.uniform(0, self.width * self.dx)
            y_mu = np.random.uniform(0, self.height * self.dy)
            x_sig = y_sig = np.random.uniform(0.1, 0.3)
            correlation = 0

            A = 1 / (2 * math.pi * x_sig * y_sig * (1 - correlation ** 2) ** 0.5)

            self.h[indices, 0, :, :] += self.params.H0 * 0.25 * torch.exp(
                - (1 / (2 * (1 - correlation ** 2))) * (
                            ((x - x_mu) / x_sig) ** 2 - 2 * correlation * ((x - x_mu) / x_sig) * ((y - y_mu) / y_sig) + (
                                (y - y_mu) / y_sig) ** 2)
            )

        # Flow velocities are zero everywhere
        self.u[indices, 0, :, :] = 0
        self.v[indices, 0, :, :] = 0

        # # Sedimentary elevation is zero everywhere
        # self.S[indices, 0, :, :] = 0
        #
        # # Vegetation density is zero everywhere, except for some randomly placed tussocks
        # mask = torch.zeros_like(self.B, dtype=torch.bool)
        # mask[indices, 0] = torch.rand(len(indices), self.height, self.width) < self.params.pEst
        # self.B[mask] = self.params.k

        # Reset time-tracking for each selected environment
        self.t[indices] = 0

    def update(self, indices):
        """
        Update selected environments
        """
        
        # Boundary condition masks and conditions
        self.cond_mask[indices, 0, :, :] = 0
        self.cond_mask[indices, 0, 0, :] = 1
        self.cond_mask[indices, 0, -1, :] = 1
        self.cond_mask[indices, 0, :, 0] = 1
        self.cond_mask[indices, 0, :, -1] = 1

        # All closed boundaries
        self.h_cond[indices, 0, 0, :] = self.h[indices, 0, 1, :]        # Top
        self.h_cond[indices, 0, -1, :] = self.h[indices, 0, -2, :]      # Bottom
        self.h_cond[indices, 0, :, 0] = self.h[indices, 0, :, 1]        # Left
        self.h_cond[indices, 0, :, -1] = self.h[indices, 0, :, -2]      # Right

        self.u_cond[indices, 0, 0, :] = self.u[indices, 0, 1, :]
        self.u_cond[indices, 0, -1, :] = self.u[indices, 0, -2, :]
        self.u_cond[indices, 0, :, 0] = -self.u[indices, 0, :, 1]
        self.u_cond[indices, 0, :, -1] = -self.u[indices, 0, :, -2]

        self.v_cond[indices, 0, 0, :] = -self.v[indices, 0, 1, :]
        self.v_cond[indices, 0, -1, :] = -self.v[indices, 0, -2, :]
        self.v_cond[indices, 0, :, 0] = self.v[indices, 0, :, 1]
        self.v_cond[indices, 0, :, -1] = self.v[indices, 0, :, -2]


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
                self.cond_mask[self.asked_indices].to(self.device), \
                self.h_cond[self.asked_indices].to(self.device), \
                self.u_cond[self.asked_indices].to(self.device), \
                self.v_cond[self.asked_indices].to(self.device), \
                # self.S[self.asked_indices].to(self.device), \
                # self.B[self.asked_indices].to(self.device)

    def tell(self, h, u, v, random_reset=False):
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
        # self.S[self.asked_indices] = S.detach().cpu()
        # self.B[self.asked_indices] = B.detach().cpu()

        # Update time-tracking
        self.t[self.asked_indices] += self.params.dt
        self.average_sequence_t += 1

        if random_reset:
            if self.average_sequence_t % (self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
                self.reset(int(self.average_sequence_i))
                self.average_sequence_i = (self.average_sequence_i+1)%self.dataset_size
                print("Resetting environment!")

