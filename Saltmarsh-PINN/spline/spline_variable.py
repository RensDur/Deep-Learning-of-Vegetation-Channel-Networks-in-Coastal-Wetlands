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

    def to(self, torch_device):
        self.device = torch_device

        # Move data to the new device
        self.offset_summary = self.offset_summary.to(self.device)

    def get_name(self):
        return self.name

    def hidden_size(self) -> int:
        return np.prod([i+1 for i in self.orders])
    
    def interpolate_at(self, hidden_state, sample_points, include_derivative=False, include_laplacian=False):
        """
        :hidden_state: Spline-weights - size: bs x (orders[0]+1) * (orders[1]+1) x H x W
        :sample_points: Set of sampling points per environment in the batch - size: bs x N x 2
        :return: Interpolated values (function values, derivatives and laplacians optional) for this spline variable
        """
        
        # hidden_state contains the hidden state for this SplineVariable only (batch x (order x order) x H x W)
        # offsets contains an (N x 3) for each environment in the batch (batch x N x 2) for position (x,y)

        # Extract the number of environments in the batch
        batch_size = hidden_state.shape[0]

        # Extract the number of samples per environment
        num_samples = sample_points.shape[1]

        # The number of sample channels describes the number of values we interpolate per sample
        # func_val + dx + dy + laplace = 4
        num_sample_channels = 1 + \
            (2 if include_derivative else 0) + \
            (1 if include_laplacian else 0)

        # The result of this interpolation is an outcome (batch x N), separately for
        # - function value
        # - derivative
        # - laplacian
        result = torch.zeros(batch_size, num_samples, num_sample_channels, device=self.device)

        sample_points = sample_points.requires_grad_(True)

        for b in range(batch_size):

            START_TIME = time.time()

            offsets = sample_points[b]

            # Extract the fractional part of each sample
            local_offsets = torch.frac(offsets).to(self.device)

            # Obtain local offsets relative to each support point of this cell
            #
            #  o--o
            #  |  |
            #  o--o
            #
            # Within each cell, we need offsets relative to each support point
            local_offsets_per_sp = local_offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,2,2) - self.offset_summary.repeat(num_samples, 1, 1, 1)

            # Repeat the offsets for each order
            local_offsets_per_sp_orders = local_offsets_per_sp.unsqueeze(2).unsqueeze(3).repeat(1,1,(self.orders[0]+1),(self.orders[1]+1),1,1)

            # Use the offsets to obtain function values for each spline kernel
            sample_kernels = torch.zeros(num_samples, num_sample_channels, (self.orders[0]+1), (self.orders[1]+1), 2, 2).to(self.device)
            for l in range(self.orders[0]+1):
                for m in range(self.orders[1]+1):
                    # Function value (directy from linear combination of splines)
                    sample_kernels[torch.arange(num_samples),0:1,l,m,:,:] = kernels.p_multidim(local_offsets_per_sp_orders[torch.arange(num_samples),:,l,m],[self.orders[0],self.orders[1]],[l,m])

            if include_derivative:
                sample_kernels[torch.arange(num_samples),1:3] = operators.grad(sample_kernels[torch.arange(num_samples),0:1], local_offsets_per_sp_orders, create_graph=True, retain_graph=True)
                
            if include_derivative and include_laplacian:
                sample_kernels[torch.arange(num_samples),3:4] = operators.div(sample_kernels[torch.arange(num_samples),1:3], local_offsets_per_sp_orders, retain_graph=True)

            # Cast the local evaluations to the right shape, grouping orders in one dimension
            sample_kernels = sample_kernels.reshape(num_samples, num_sample_channels, (self.orders[0]+1)*(self.orders[1]+1), 2, 2)

            # Round down to obtain top-left support point indices
            top_left_support_point = torch.floor(offsets).int()

            tx = top_left_support_point[:, 0]
            ty = top_left_support_point[:, 1]

            # Extract local support point weights for each sample
            support_00 = hidden_state[b, :, ty, tx].T # Top left support points - shape [#samples, lxm]
            support_01 = hidden_state[b, :, ty, tx+1].T # Top right support points - shape [#samples, lxm]
            support_10 = hidden_state[b, :, ty+1, tx].T # Bottom left support points - shape [#samples, lxm]
            support_11 = hidden_state[b, :, ty+1, tx+1].T # Bottom right support points - shape [#samples, lxm]

            # Arrange the support point weights
            hidden_patch = torch.stack([
                torch.stack([support_00, support_01], dim=-1),
                torch.stack([support_10, support_11], dim=-1)
            ], dim=-1)  # Shape [#samples, lxm, 2, 2]

            hidden_patch = hidden_patch.unsqueeze(1).repeat(1, num_sample_channels, 1, 1, 1)

            hidden_patch = hidden_patch.reshape(num_samples * num_sample_channels, (self.orders[0]+1)*(self.orders[1]+1), 2, 2)
            sample_kernels = sample_kernels.reshape(num_samples * num_sample_channels, (self.orders[0]+1)*(self.orders[1]+1), 2, 2)

            # Multiply the weights with the kernels and sum the spline kernels per sample
            result[b, ...] = (sample_kernels * hidden_patch).sum(dim=(1, 2, 3)).reshape(num_samples, num_sample_channels)

            DURATION = time.time() - START_TIME

            print(f"Processed env {b}/{batch_size} in {DURATION}s")


        return result
    

    def interpolate_superres_at(self, weights, resolution_factor):

        res_key = f"{resolution_factor}, orders: {self.orders}"
        
        if res_key in self.kernel_buffer_superres.keys():
            self.superres_kernels = self.kernel_buffer_superres[res_key]
        else:
            self.superres_kernels = torch.zeros(1,self.kernel_size,(self.orders[0]+1)*(self.orders[1]+1),2*resolution_factor,2*resolution_factor).to(self.device)

            for i in range(resolution_factor):
                for j in range(resolution_factor):
                    offsets = torch.tensor([i/resolution_factor,j/resolution_factor], device=self.device).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)-1 + self.offset_summary
                    offsets = offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,(self.orders[0]+1),(self.orders[1]+1),1,1).detach().requires_grad_(True)
                    
                    sub_kernels = torch.zeros(1,self.kernel_size,(self.orders[0]+1),(self.orders[1]+1),2,2, device=self.device)
                    for l in range(self.orders[0]+1):
                        for m in range(self.orders[1]+1):
                            # Function value (directy from linear combination of splines)
                            sub_kernels[0:1,0:1,l,m,:,:] = kernels.p_multidim(offsets[:,:,l,m],[self.orders[0],self.orders[1]],[l,m])

                    
                    # First derivative (d/dx and d/dy)
                    if self.requires_derivative:
                        sub_kernels[0:1,1:3] = operators.grad(sub_kernels[0:1,0:1,:,:,:,:],offsets,create_graph=True,retain_graph=True)

                    # Laplace -- Note: laplacian without first derivative is not supported (quicker computation)
                    if self.requires_laplacian:
                        sub_kernels[0:1,3:4] = operators.div(sub_kernels[0:1,1:3], offsets, retain_graph=False)
                    
                    sub_kernels = sub_kernels.reshape(1,self.kernel_size,(self.orders[0]+1)*(self.orders[1]+1),2,2).detach()
                    self.superres_kernels[:,:,:,i::resolution_factor,j::resolution_factor] = sub_kernels

            # buffer kernels
            self.superres_kernels = self.superres_kernels.permute(0,2,1,3,4)
            self.kernel_buffer_superres[res_key] = self.superres_kernels
            self.save_buffers()

        output = F.conv_transpose2d(weights,self.superres_kernels[0],padding=0,stride=resolution_factor)

        return output[:, 0:1], \
                output[:, 1:3] if self.requires_derivative else None, \
                output[:, 3:4] if self.requires_laplacian else None
        
    


var = SplineVariable('f', 1, torch.device("mps"))
hidden_state = torch.zeros(50, var.hidden_size(), 101, 101, device=torch.device("mps"))
sample_points = torch.rand(50, 10000, 2, device=torch.device("mps")) * 100

