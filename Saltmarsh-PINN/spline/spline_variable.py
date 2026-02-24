import os,pickle
import torch
import torch.nn.functional as F
import numpy as np
import spline.kernels as kernels
import spline.operators as operators
import time

def _dbg(_desc='',_expr=None):
    print(f"DBG! >> {_desc}:\n{_expr}")
    return _expr

def _dbg_nan(_desc='',_tensor=torch.zeros(1)):
    _expr = torch.mean(_tensor).detach().cpu()
    _dbg(_desc, _expr)
    return _tensor

def _dbg_blocking(_desc='',_expr=None):
    _ = _dbg(_desc, _expr)
    _ = input("HIT ENTER TO STEP")
    return _expr

class SplineVariable:

    def __init__(self, name, order: int, device=torch.device("cpu")):

        # Name for buffering on disk
        self.name = name

        # Spline order = (degree polynomial) + 1
        self.orders = [order, order]

        # Torch device
        self.device = device

        # Prepare the required spline kernels for this variable
        self.offset_summary = torch.tensor([[[0,0],[1,0]],[[0,1],[1,1]]]).unsqueeze(0).permute(0,3,2,1).to(self.device)

        # Caching kernels
        self.kernel_buffer = {}

    def to(self, torch_device):
        self.device = torch_device

        # Move data to the new device
        self.offset_summary = self.offset_summary.to(self.device)

    def get_name(self):
        return self.name

    def hidden_size(self) -> int:
        return np.prod([i+1 for i in self.orders])

    def interpolate_at_regular_interval(self, hidden_state, offset, include_derivative=False, include_laplacian=False):
        """
        Create an image (snapshot) of the spline field by interpolating at the same offset within each cell of the domain.
        Every cell is surrounded by 4 support points. This method resembles the implementation by Wandel et. al. [TODO: CHECK REF.]
        :hidden_state: Spline-weights - size: bs x (orders[0]+1) * (orders[1]+1) x H+1 x W+1
        :offset: Offset at which to sample within each cell - size: 2:{x,y}
        :return: Interpolated image with regular intervals between sampling points - size: bs x #channels x H x W
        """

        # The number of sample channels describes the number of values we interpolate per sample
        # func_val + dx + dy + laplace = 4
        num_sample_channels = 1 + \
            (2 if include_derivative else 0) + \
            (1 if include_laplacian else 0)

        # ID this specific kernel by offset {x,y}, spline orders and whether or not derivatives or laplacian are included
        offset_key = f"{offsets[0]} {offsets[1]}, orders: {self.orders}, deriv: {include_derivative}, laplace: {include_laplacian}"

        if offset_key in self.kernel_buffer.keys():
            sample_kernels = self.kernel_buffer[offset_key]
        else:
        
            # The incoming offset is a local coordinate (dx, dy) describing a position between four support points
            # Repeat the offset four times and organise them in a grid of shape [1,   2, 2, 2]
            #                                                                   [_, x/y, <[0,0],[0,1],[1,0],[1,1]>]
            # This gives an offset coordinate relative to each of the four corners in a cell.
            offsets = (offsets.clone().unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)-self.offset_summary)

            # Repeat the offsets for each order
            offsets = offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,(self.orders[0]+1),(self.orders[1]+1),1,1).detach().requires_grad_(True)

            # Prepare the new kernels for these offsets
            sample_kernels = torch.zeros(1, num_sample_channels, (self.orders[0]+1), (self.orders[1]+1), 2, 2).to(self.device)
            for l in range(self.orders[0]+1):
                for m in range(self.orders[1]+1):
                    # Function value (directy from linear combination of splines)
                    sample_kernels[0:1,0:1,l,m,:,:] = kernels.p_multidim(offsets[:,:,l,m],[self.orders[0],self.orders[1]],[l,m])

            # First derivative (d/dx and d/dy)
            if include_derivative:
                sample_kernels[0:1,1:3] = operators.grad(sample_kernels[0:1,0:1,:,:,:,:],offsets,create_graph=True,retain_graph=True)

            # Laplace -- Note: laplacian without first derivative is not supported (quicker computation)
            if include_derivative and include_laplacian:
                sample_kernels[0:1,3:4] = operators.div(sample_kernels[0:1,1:3], offsets, retain_graph=True)

            sample_kernels = sample_kernels.reshape(1, num_sample_channels, (self.orders[0]+1)*(self.orders[1]+1), 2, 2).detach() # Group orders in one channel

            #
            # Save this kernel in cache
            #
            self.kernel_buffer[offset_key] = sample_kernels

        output = F.conv2d(hidden_state, sample_kernels[0], padding=0)

        return output



    
    def interpolate_at(self, hidden_state, sample_points, include_derivative=False, include_laplacian=False):
        """
        :hidden_state: Spline-weights - size: bs x (orders[0]+1) * (orders[1]+1) x H+1 x W+1
        :sample_points: Set of sampling points per environment in the batch - size: bs x N x 2
        :return: Interpolated values (function values, derivatives and laplacians optional) for this spline variable
                 shape: (bs x num_samples x num_sample_channels)
        """
        
        # hidden_state contains the hidden state for this SplineVariable only (batch x (order x order) x H x W)
        # offsets contains an (N x 3) for each environment in the batch (batch x N x 2) for position (x,y)

        # Extract the number of environments in the batch
        batch_size = hidden_state.shape[0]

        # Extract the number of samples per environment
        num_samples = sample_points.shape[1]

        # Total number of samples
        # This is used to 'batch over the total number of samples', instead of first batching
        # over the various environments and then over the samples inside that environment
        total_num_samples = batch_size * num_samples

        # The number of sample channels describes the number of values we interpolate per sample
        # func_val + dx + dy + laplace = 4
        num_sample_channels = 1 + \
            (2 if include_derivative else 0) + \
            (1 if include_laplacian else 0)

        # The result of this interpolation is an outcome (batch x N), separately for
        # - function value
        # - derivative
        # - laplacian
        sample_points = sample_points.requires_grad_(True)

        #
        # GPU ACCELERATED SAMPLING
        #

        # Reshape the offsets to group batch and num_samples
        sample_points = sample_points.reshape(total_num_samples, 2)

        # Extract the fractional part of each sample
        local_offsets = torch.frac(sample_points).to(self.device)

        # Obtain local offsets relative to each support point of this cell
        #
        #  o--o
        #  |  |
        #  o--o
        #
        # Within each cell, we need offsets relative to each support point
        local_offsets_per_sp = local_offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,2,2) - self.offset_summary.repeat(total_num_samples, 1, 1, 1)

        # Repeat the offsets for each order
        local_offsets_per_sp_orders = local_offsets_per_sp.unsqueeze(2).unsqueeze(3).repeat(1,1,(self.orders[0]+1),(self.orders[1]+1),1,1)

        # Use the offsets to obtain function values for each spline kernel
        sample_kernels = torch.zeros(total_num_samples, num_sample_channels, (self.orders[0]+1), (self.orders[1]+1), 2, 2).to(self.device)
        for l in range(self.orders[0]+1):
            for m in range(self.orders[1]+1):
                # Function value (directy from linear combination of splines)
                sample_kernels[torch.arange(total_num_samples),0:1,l,m,:,:] = kernels.p_multidim(local_offsets_per_sp_orders[torch.arange(total_num_samples),:,l,m],[self.orders[0],self.orders[1]],[l,m])

        if include_derivative:
            sample_kernels[torch.arange(total_num_samples),1:3] = operators.grad(sample_kernels[torch.arange(total_num_samples),0:1], local_offsets_per_sp_orders, create_graph=True, retain_graph=True)
            
        if include_derivative and include_laplacian:
            sample_kernels[torch.arange(total_num_samples),3:4] = operators.div(sample_kernels[torch.arange(total_num_samples),1:3], local_offsets_per_sp_orders, retain_graph=True)

        # Cast the local evaluations to the right shape, grouping orders in one dimension
        sample_kernels = sample_kernels.reshape(total_num_samples, num_sample_channels, (self.orders[0]+1)*(self.orders[1]+1), 2, 2)

        # Round down to obtain top-left support point indices
        top_left_support_point = torch.floor(sample_points).int()

        tx = top_left_support_point[:, 0]
        ty = top_left_support_point[:, 1]

        # Batch index for each flattened sample: sample k belongs to batch k // num_samples
        batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, num_samples).reshape(-1)

        # Extract local support point weights for each sample -> (total_num_samples, lxm)
        support_00 = hidden_state[batch_idx, :, ty, tx]
        support_01 = hidden_state[batch_idx, :, ty, tx + 1]
        support_10 = hidden_state[batch_idx, :, ty + 1, tx]
        support_11 = hidden_state[batch_idx, :, ty + 1, tx + 1]

        # Arrange the support point weights
        hidden_patch = torch.stack([
            torch.stack([support_00, support_01], dim=-1),
            torch.stack([support_10, support_11], dim=-1)
        ], dim=-1)  # Shape [#samples, lxm, 2, 2]

        hidden_patch = hidden_patch.unsqueeze(1).repeat(1, num_sample_channels, 1, 1, 1)

        result = (sample_kernels * hidden_patch).sum(dim=(2, 3, 4)).reshape(batch_size, num_samples, num_sample_channels)

        return result
    

    def interpolate_highres(self, hidden_state, width, height, include_derivative=False, include_laplacian=False):

        batch_size = hidden_state.shape[0]

        spline_width = hidden_state.shape[3]
        spline_height = hidden_state.shape[2]

        xs = torch.arange(width, device=self.device) / width * (spline_width-1) + 0.5 / width * (spline_width-1)
        ys = torch.arange(height, device=self.device) / height * (spline_height-1) + 0.5 / height * (spline_height-1)

        x_grid, y_grid = torch.meshgrid(xs, ys, indexing='xy')

        sample_points = torch.stack([x_grid, y_grid], dim=-1).reshape(width*height, 2).unsqueeze(0).repeat(batch_size, 1, 1)

        # Interpolate at these sample points to obtain an image
        interpolated_samples = self.interpolate_at(hidden_state, sample_points, include_derivative, include_laplacian)

        # Reshape to obtain an image tensor
        num_sample_channels = interpolated_samples.shape[2]

        image = interpolated_samples.reshape(batch_size, height, width, num_sample_channels).swapdims(1, 3).swapdims(2, 3)

        return image

        
        
    
width = 5
height = 5

var = SplineVariable('f', 1, torch.device("cpu"))
hidden_state = torch.zeros(1, var.hidden_size(), height+1, width+1, device=torch.device("cpu"))


k = var.interpolate_highres(hidden_state, 1920, 1080)