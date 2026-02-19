import torch
import torch.nn.functional as F
import numpy as np
import math
import kernels
import multiprocessing

# Find the number of available CPUs, capped at 8
NUM_CPUS = multiprocessing.cpu_count()
torch.set_num_threads(NUM_CPUS)
print(f"Using {NUM_CPUS} threads")

class Testset:

    def __init__(self):

        self.width = 200
        self.height = 200

        self.device = torch.device("mps")

        self.orders = [1, 1]

        # Hidden state
        self.hidden_states = torch.zeros(
            1,
            self.hidden_size(),
            self.height+1,
            self.width+1,
            device=self.device
        )

        self.hidden_states[:, 0, :, :] = 200

        self.kernel_size = 1

        self.offset_summary = torch.tensor([[[0,0],[1,0]],[[0,1],[1,1]]]).unsqueeze(0).permute(0,3,2,1).to(self.device)


    def hidden_size(self) -> int:
        return np.prod([i+1 for i in self.orders])  

    def interpolate_sample(self, offset):

        # offset in [0, width], [0, height]
        local_offset = torch.frac(offset).to(self.device) # Retrieve fractional part == (dx, dy) within local cell

        # Obtain local offset relative to all support points of this cell
        # (four corners)
        local_offset_corners = local_offset.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2) - self.offset_summary

        # Repeat this offset for each order
        local_offset_corners_orders = local_offset_corners.unsqueeze(2).unsqueeze(3).repeat(1,1,(self.orders[0]+1),(self.orders[1]+1),1,1)

        # Prepare the new kernels for these offsets
        self.kernels = torch.zeros(1, self.kernel_size, (self.orders[0]+1), (self.orders[1]+1), 2, 2).to(self.device)
        for l in range(self.orders[0]+1):
            for m in range(self.orders[1]+1):
                # Function value (directy from linear combination of splines)
                self.kernels[0:1,0:1,l,m,:,:] = kernels.p_multidim(local_offset_corners_orders[:,:,l,m],[self.orders[0],self.orders[1]],[l,m])

        # Multiplicant
        multiplicant = torch.zeros(1, self.kernel_size, (self.orders[0]+1), (self.orders[1]+1), self.height+1, self.width+1).to(self.device)

        top_left_support_point = torch.floor(offset).int()
        top_left_x = top_left_support_point[0]
        top_left_y = top_left_support_point[1]

        multiplicant[..., top_left_y:top_left_y+2, top_left_x:top_left_x+2] = self.kernels

        # Reshape to align all order parts
        multiplicant = multiplicant.reshape(1, (self.orders[0]+1)*(self.orders[1]+1), self.height+1, self.width+1)

        # Convolution
        out = F.conv2d(multiplicant, self.hidden_states, padding=0)

        return out

    def interpolate_multiple_samples(self, offsets):

        # Offsets of shape (#samples(N), 2)
        num_samples = offsets.shape[0]

        # Grab the fractional part
        local_offsets = torch.frac(offsets).to(self.device)

        # Obtain local offset relative to all support points of this cell
        # (four corners)
        local_offsets_corners = local_offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,2,2) - self.offset_summary.repeat(num_samples, 1, 1, 1)

        # Repeat this offset for each order
        local_offsets_corners_orders = local_offsets_corners.unsqueeze(2).unsqueeze(3).repeat(1,1,(self.orders[0]+1),(self.orders[1]+1),1,1)

        # Prepare the new kernels for these offsets
        self.kernels = torch.zeros(num_samples, self.kernel_size, (self.orders[0]+1), (self.orders[1]+1), 2, 2).to(self.device)
        for l in range(self.orders[0]+1):
            for m in range(self.orders[1]+1):
                # Function value (directy from linear combination of splines)
                self.kernels[torch.arange(num_samples),0:1,l,m,:,:] = kernels.p_multidim(local_offsets_corners_orders[torch.arange(num_samples),:,l,m],[self.orders[0],self.orders[1]],[l,m])

        self.kernels = self.kernels.reshape(num_samples, (self.orders[0]+1)*(self.orders[1]+1), 2, 2)

        # Round down to obtain top-left support point indices
        top_left_support_point = torch.floor(offsets).int()

        tx = top_left_support_point[:, 0]
        ty = top_left_support_point[:, 1]

        # Extract local support point weights for each sample
        support_00 = self.hidden_states[0, :, ty, tx].T # Top left support points - shape [#samples, lxm]
        support_01 = self.hidden_states[0, :, ty, tx+1].T # Top right support points - shape [#samples, lxm]
        support_10 = self.hidden_states[0, :, ty+1, tx].T # Bottom left support points - shape [#samples, lxm]
        support_11 = self.hidden_states[0, :, ty+1, tx+1].T # Bottom right support points - shape [#samples, lxm]

        # Arrange the support point weights
        hidden_patch = torch.stack([
            torch.stack([support_00, support_01], dim=-1),
            torch.stack([support_10, support_11], dim=-1)
        ], dim=-1)  # Shape [#samples, lxm, 2, 2]

        # Multiply the weights with the kernels and sum the spline kernels per sample
        out = (self.kernels * hidden_patch).sum(dim=(1, 2, 3)).reshape(num_samples)

        return out



def main():

    offset = torch.rand(2) # Random (dx, dy) coordinate
    print(offset)




if __name__ == "__main__":
    main()

testset = Testset()
k = testset.interpolate_multiple_samples(torch.rand(100, 2) * testset.width)