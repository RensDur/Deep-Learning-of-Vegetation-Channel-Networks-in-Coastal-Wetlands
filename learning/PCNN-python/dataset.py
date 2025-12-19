import torch
import numpy as np
import math



class Dataset:
    def __init__(self, params, device):

        # Dimensions
        self.width = params.width
        self.height = params.height
        self.padding = 5
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
        self.cond_mask = torch.zeros(self.dataset_size, 1, self.height, self.width).int() # Boundary mask - [1 - Omega] - 1 on the boundaries, 0 everywhere else
        self.flux_x_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)
        self.flux_y_cond = torch.zeros(self.dataset_size, 1, self.height, self.width)

        # Time tracking per environment
        self.t = torch.zeros(self.dataset_size)

        self.average_sequence_length = params.average_sequence_length
        self.average_sequence_t = 0
        self.average_sequence_i = 0
        self.num_full_resets = 0

        # Reset all environments upon initialization
        self.reset(range(self.dataset_size))

        # Preselect pytorch device
        self.device = device


    def reset(self, indices):
        """
        METHOD: Reset environments
        """

        # Uniform water thickness H0
        self.h[indices, 0, :, :] = 2

        # wavespan = int(self.width / 10)
        # x = torch.linspace(0, wavespan, wavespan)

        # self.h[indices, 0, :, :wavespan] += 0.2 * torch.cos((x / wavespan) * 0.5 * math.pi)

        x = torch.linspace(0, self.width * self.dx, self.width)
        y = torch.linspace(0, self.height * self.dy, self.height)
        x, y = torch.meshgrid(x, y, indexing='xy')

        for _ in range(1):
            x_mu = np.random.uniform(0, self.width * self.dx)
            y_mu = np.random.uniform(0, self.height * self.dy)
            x_sig = y_sig = 1
            correlation = 0

            A = 1 / (2 * math.pi * x_sig * y_sig * (1 - correlation ** 2) ** 0.5)

            self.h[indices, 0, :, :] += 0.2 * torch.exp(
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
        self.cond_mask[indices, 0, :, :] = 1
        self.cond_mask[indices, 0, self.padding:-self.padding, self.padding:-self.padding] = 0

        # Add rounded corners to the cond_mask
        # circle = torch.zeros(int(self.width/4), int(self.height/4), dtype=torch.int)
        # diameter = circle.shape[0]/2
        # cx = int(circle.shape[0]/2)
        # cy = int(circle.shape[1]/2)

        # for x in range(circle.shape[0]):
        #     for y in range(circle.shape[1]):
        #         dx2 = (x - cx)**2
        #         dy2 = (y - cy)**2
        #         if dx2 + dy2 >= diameter**2:
        #             circle[y, x] = 1

        # self.cond_mask[indices, 0, self.padding:self.padding+cy, self.padding:self.padding+cx] = circle[:cy, :cx]
        # self.cond_mask[indices, 0, self.padding:self.padding+cy, -cx-self.padding:-self.padding] = circle[:cy, -cx:]
        # self.cond_mask[indices, 0, -cy-self.padding:-self.padding, self.padding:self.padding+cx] = circle[-cy:, :cx]
        # self.cond_mask[indices, 0, -cy-self.padding:-self.padding, -cx-self.padding:-self.padding] = circle[-cy:, -cx:]

        if self.padding == 0:
            self.cond_mask[indices, 0, :, :] = 0

        # All closed boundaries
        self.flux_x_cond[indices, 0, :, :] = 0
        self.flux_y_cond[indices, 0, :, :] = 0



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
                self.flux_x_cond[self.asked_indices].to(self.device), \
                self.flux_y_cond[self.asked_indices].to(self.device), \
                # self.S[self.asked_indices].to(self.device), \
                # self.B[self.asked_indices].to(self.device)

    def tell(self, h, u, v, batch_loss, random_reset=False):
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

        # if random_reset:
        #     if self.average_sequence_t % (self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
        #         self.reset(int(self.average_sequence_i))
        #         self.average_sequence_i = (self.average_sequence_i+1)%self.dataset_size
        #         print("Resetting environment!")

            # if self.average_sequence_t % (10*int(1 + self.num_full_resets)*self.average_sequence_length/self.batch_size) == 0:
            #     # Reset the entire dataset every now and then to prevent training on unphysical data
            #     self.reset(range(self.dataset_size))
            #     self.num_full_resets += 1
            #     self.average_sequence_t = 0
            #     print("Resetting entire dataset!")

        # We will reset the 10% most 'unphysical' environments in this batch
        # This is measured by the loss per environment, given by the batch_loss
        num_to_reset = max(1, int(self.batch_size * 0.1))
        largest_loss = torch.topk(batch_loss, num_to_reset)
        
        # Reset all these unphysical environments
        self.reset(largest_loss.indices.cpu())
