import os,pickle
import torch
import torch.nn.functional as F
import numpy as np
import spline.kernels as kernels
import spline.operators as operators

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

    def __init__(self, name, order: int, requires_derivative=False, requires_laplacian=False, device=torch.device("cpu")):

        # Name for buffering on disk
        self.name = name

        # Spline order = (degree polynomial) + 1
        self.orders = [2, order, order]

        # Torch device
        self.device = device

        # Prepare the required spline kernels for this variable
        self.requires_derivative = requires_derivative or requires_laplacian # Requiring laplacian without requiring first.deriv. is unsupported.
        self.requires_laplacian = requires_laplacian
        self.kernel_size = 1 \
            + (3 if self.requires_derivative else 0) \
            + (1 if self.requires_laplacian else 0)

        self.offset_summary = torch.tensor( [[[[[0, 0],         # T-COORDINATES
                                                [0, 0]],

                                               [[1, 1],
                                                [1, 1]]],


                                              [[[0, 0],         # Y-COORDINATES
                                                [1, 1]],

                                               [[0, 0],
                                                [1, 1]]],


                                              [[[0, 1],         # X-COORDINATES
                                                [0, 1]],

                                               [[0, 1],
                                                [0, 1]]]]]).to(self.device)
        self.kernel_buffer = {}
        self.kernel_buffer_superres = {}

        # Immediately try to load the buffers for this variable
        try:
            self.load_buffers()
            print(f"Loaded buffers for SplineVariable '{self.name}'")
        except:
            print(f"No buffers available for SplineVariable '{self.name}'")

    def to(self, torch_device):
        self.device = torch_device

        # Move data to the new device
        self.offset_summary = self.offset_summary.to(self.device)

        for k in self.kernel_buffer.keys():
            self.kernel_buffer[k] = self.kernel_buffer[k].to(self.device)

        for k in self.kernel_buffer_superres.keys():
            self.kernel_buffer_superres[k] = self.kernel_buffer_superres[k].to(self.device)

    def get_name(self):
        return self.name

    def hidden_size(self) -> int:
        return np.prod([i+1 for i in self.orders])

    def save_buffers(self):
        os.makedirs("Logger/spline_kernel_buffers",exist_ok=True)
        path = f"Logger/spline_kernel_buffers/kernel_buffers_{self.name}.dic"
        with open(path,"wb") as file:
            pickle.dump({"kernel_buffer":self.kernel_buffer,"kernel_buffer_superres":self.kernel_buffer_superres}, file)

    def load_buffers(self):
        path = f"Logger/spline_kernel_buffers/kernel_buffers_{self.name}.dic"
        with open(path,"rb") as file:
            buffers = pickle.load(file)
            self.kernel_buffer = buffers["kernel_buffer"]
            self.kernel_buffer_superres = buffers["kernel_buffer_superres"]

        # Make sure these kernel buffers are all on the desired device
        self.to(self.device)
    
    def interpolate_at(self, old_hidden_state, new_hidden_state, offsets):
        """
        Idea: return derivatives of splines directly, implement with convolutions
        :weights: size: bs x (orders[0]+1) * (orders[1]+1) x w x h
        :offsets: offsets to interpolate in between weights, size: 3
        :orders: orders of spline for each dimension (note: counting starts at 0 => 0 ~ 1st order, 1 ~ 2nd order, 2 ~ 3rd order)
        :return: a_z,v,grad_v,laplace_v - note that, width / height is decreased by 1, because we only interpolate in between support points (weights)
            :a_z: vector potential of velocity field, size: bs x 1 x (w-1) x (h-1)
            :rot(a_z): velocity field, size: bs x 2 x (w-1) x (h-1)
            :grad(rot(a_z)): gradient (jacobian) of velocity field (dvx/dx dvx/dy dvy/dx dvy/dy), size: bs x 4 x (w-1) x (h-1)
            :laplace(rot(a_z)): laplacian of velocity field (laplace(vx) laplace(vy)), size:  bs x 2 x (w-1) x (h-1)
        """
        # construct kernel matrix for 2x2 convolution based on offset:
        # => number of input channels = (orders[0]+1) * (orders[1]+1)
        # => number of output channels = 1 + 2 + 4 + 2 (a_z,v=rot(a_z),grad(v_x),grad(v_y),laplace(v_x),laplace(v_y)
        offset_key = f"{offsets[0]} {offsets[1]} {offsets[2]}, orders: {self.orders}, requires_derivative: {self.requires_derivative}, requires_laplacian: {self.requires_laplacian}"

        if offset_key in self.kernel_buffer.keys():
            self.kernels = self.kernel_buffer[offset_key]
        else:

            # The incoming offset is a local coordinate (dt, dy, dx) describing a position between four support points
            # Repeat the offset four times and organise them in a grid of shape [1,   2, 2, 2]
            #                                                                   [_, x/y, <[0,0],[0,1],[1,0],[1,1]>]
            #                                                                   [_, x/y/t, <[0,0,0],[0,0,1],[0,1,0],...,[1,1,1]>]
            # This gives an offset coordinate relative to each of the four corners in a cell.
            offsets = (offsets.clone().unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,2,2,2)-self.offset_summary)

            # Repeat the offsets for each order
            offsets = offsets.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,(self.orders[0]+1), (self.orders[1]+1), (self.orders[2]+1),1,1,1).detach().requires_grad_(True)
            
            # Prepare the new kernels for these offsets
            self.kernels = torch.zeros(1, self.kernel_size, (self.orders[0]+1), (self.orders[1]+1), (self.orders[2]+1), 2, 2, 2).to(self.device)
            for k in range(self.orders[0]+1): # Time order
                for l in range(self.orders[1]+1): # Space order (y)
                    for m in range(self.orders[2]+1): # Space order (x)
                        # Function value (directy from linear combination of splines)
                        self.kernels[0:1,0:1,k,l,m,:,:,:] = kernels.p_multidim(offsets[:,:,k,l,m],[self.orders[0],self.orders[1],self.orders[2]],[k,l,m])
            
            # First derivative (d/dt | d/dy | d/dx)
            if self.requires_derivative:
                self.kernels[0:1,1:4] = operators.grad(self.kernels[0:1,0:1,:,:,:,:,:,:],offsets,create_graph=True,retain_graph=True)

            # Laplace -- Note: laplacian without first derivative is not supported (quicker computation)
            if self.requires_laplacian:
                self.kernels[0:1,4:5] = operators.div(self.kernels[0:1,2:4], offsets, retain_graph=True)
            
            self.kernels = self.kernels.reshape(1, self.kernel_size, (self.orders[0]+1)*(self.orders[1]+1)*(self.orders[2]+1), 2, 2, 2).detach() # Group orders in one channel
            
            # buffer self.kernels
            self.kernel_buffer[offset_key] = self.kernels
            self.save_buffers()

        # The weights for this convolution are now the two hidden states from different points in time stacked on top of eachother
        weights = torch.stack([old_hidden_state, new_hidden_state], dim=2)

        output = F.conv3d(weights,self.kernels[0],padding=0).squeeze(2) # By squeeze(2), we squeeze the time dimension to end up with an output of shape (batch_size, C, H, W)

        return output[:, 0:1], \
                output[:, 1:4] if self.requires_derivative else None, \
                output[:, 4:5] if self.requires_laplacian else None
    

    def interpolate_superres_at(self, weights, resolution_factor):

        res_key = f"{resolution_factor}, orders: {self.orders}"
        
        if res_key in self.kernel_buffer_superres.keys():
            self.superres_kernels = self.kernel_buffer_superres[res_key]
        else:
            self.superres_kernels = torch.zeros(1,self.kernel_size,(self.orders[0]+1)*(self.orders[1]+1)*(self.orders[2]+1),2,2*resolution_factor,2*resolution_factor).to(self.device)

            for i in range(resolution_factor):
                for j in range(resolution_factor):
                    offsets = torch.tensor([0.5, i/resolution_factor,j/resolution_factor], device=self.device).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,2,2,2)-1 + self.offset_summary
                    offsets = offsets.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,(self.orders[0]+1),(self.orders[1]+1),(self.orders[2]+1),1,1,1).detach().requires_grad_(True)
                    
                    sub_kernels = torch.zeros(1,self.kernel_size,(self.orders[0]+1),(self.orders[1]+1),(self.orders[2]+1),2,2,2, device=self.device)
                    for k in range(self.orders[0]+1):
                        for l in range(self.orders[1]+1):
                            for m in range(self.orders[2]+1):
                                # Function value (directy from linear combination of splines)
                                sub_kernels[0:1,0:1,k,l,m,:,:,:] = kernels.p_multidim(offsets[:,:,k,l,m],[self.orders[0],self.orders[1],self.orders[2]],[k,l,m])

                    
                    # First derivative (d/dt, d/dy and d/dx)
                    if self.requires_derivative:
                        sub_kernels[0:1,1:4] = operators.grad(sub_kernels[0:1,0:1,:,:,:,:,:,:],offsets,create_graph=True,retain_graph=True)

                    # Laplace -- Note: laplacian without first derivative is not supported (quicker computation)
                    if self.requires_laplacian:
                        sub_kernels[0:1,4:5] = operators.div(sub_kernels[0:1,2:4], offsets, retain_graph=False)
                    
                    sub_kernels = sub_kernels.reshape(1,self.kernel_size,(self.orders[0]+1)*(self.orders[1]+1)*(self.orders[2]+1),2,2,2).detach()
                    self.superres_kernels[:,:,:,:,i::resolution_factor,j::resolution_factor] = sub_kernels

            # buffer kernels
            self.superres_kernels = self.superres_kernels.permute(0,2,1,3,4,5)
            self.kernel_buffer_superres[res_key] = self.superres_kernels
            self.save_buffers()

        output = F.conv_transpose3d(torch.stack([weights, weights], dim=2),self.superres_kernels[0],padding=0,stride=(1, resolution_factor, resolution_factor))
        
        # The transposed convolution creates two additional time-layers:
        # [0] - Dominated by the old hidden state
        # [1] - Mixed the two hidden states
        # [2] - Dominated by the new hidden state
        output = output[:, :, 1, :, :]

        return output[:, 0:1], \
                output[:, 1:4] if self.requires_derivative else None, \
                output[:, 4:5] if self.requires_laplacian else None
        
