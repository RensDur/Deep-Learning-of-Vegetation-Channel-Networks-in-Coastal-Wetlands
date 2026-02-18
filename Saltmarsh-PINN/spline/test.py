import torch
import torch.nn.functional as F
import numpy as np
import math
import kernels

class Testset:

    def __init__(self):

        self.width = 5
        self.height = 5

        self.device = torch.device("cpu")

        self.orders = [1, 1]

        # Hidden state
        self.hidden_states = torch.zeros(
            1,
            self.hidden_size(),
            self.height,
            self.width,
        )

        self.kernel_size = 1

        self.offset_summary = torch.tensor([[[0,0],[1,0]],[[0,1],[1,1]]]).unsqueeze(0).permute(0,3,2,1).to(self.device)


    def hidden_size(self) -> int:
        return np.prod([i+1 for i in self.orders])  

    def interpolate_sample(self, offset):

        # offset in [0, width], [0, height]
        local_offset = torch.frac(offset) # Retrieve fractional part == (dx, dy) within local cell

        # Obtain local offset relative to all support points of this cell
        # (four corners)
        local_offset_corners = local_offset.clone().unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2) - self.offset_summary

        # Repeat this offset for each order
        local_offset_corners_orders = local_offset_corners.unsqueeze(2).unsqueeze(3).repeat(1,1,(self.orders[0]+1),(self.orders[1]+1),1,1)

        # Prepare the new kernels for these offsets
        self.kernels = torch.zeros(1, self.kernel_size, (self.orders[0]+1), (self.orders[1]+1), 2, 2).to(self.device)
        for l in range(self.orders[0]+1):
            for m in range(self.orders[1]+1):
                # Function value (directy from linear combination of splines)
                self.kernels[0:1,0:1,l,m,:,:] = kernels.p_multidim(local_offset_corners_orders[:,:,l,m],[self.orders[0],self.orders[1]],[l,m])

        # Multiplicant
        multiplicant = torch.zeros(1, self.kernel_size, (self.orders[0]+1), (self.orders[1]+1), self.height, self.width).to(self.device)

        top_left_support_point = torch.floor(offset).int()
        top_left_x = top_left_support_point[0]
        top_left_y = top_left_support_point[1]

        multiplicant[..., top_left_y:top_left_y+2, top_left_x:top_left_x+2] = self.kernels

        # Reshape to align all order parts
        multiplicant = multiplicant.reshape(1, (self.orders[0]+1)*(self.orders[1]+1), self.height, self.width)

        # Convolution
        out = F.conv2d(multiplicant, self.hidden_states, padding=0)

        return out

    




def main():

    offset = torch.rand(2) # Random (dx, dy) coordinate
    print(offset)




if __name__ == "__main__":
    main()