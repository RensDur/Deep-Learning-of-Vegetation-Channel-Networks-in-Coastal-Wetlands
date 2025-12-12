import os,pickle
import torch
from numpy import np
import kernels

class SplineVariable:

    def __init__(self, name, order: int, device):

        # Name for buffering on disk
        self.name = name

        # Spline order = (degree polynomial) + 1
        self.orders = [order, order]

        # Torch device
        self.device = device

        # Prepare the required spline kernels for this variable
        self.kernels = torch.zeros(1,1+2+4+2,(self.orders[0]+1),(self.orders[1]+1),2,2).to(self.device)

        self.offset_summary = torch.tensor([[[0,0],[1,0]],[[0,1],[1,1]]]).unsqueeze(0).permute(0,3,2,1).to(self.device)
        self.kernel_buffer = {}
        self.kernel_buffer_superres = {}

        # Immediately try to load the buffers for this variable
        try:
            self.load_buffers()
            print(f"Loaded buffers for SplineVariable '{self.name}'")
        except:
            print(f"No buffers available for SplineVariable '{self.name}'")

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
    
    def interpolate_at(self, weights, offsets):
        """
        Idea: return derivatives of splines directly, implement with convolutions
        :weights: size: bs x (orders[0]+1) * (orders[1]+1) x w x h
        :offsets: offsets to interpolate in between weights, size: 2
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
        offset_key = f"{offsets[0]} {offsets[1]}, orders: {orders}"
        if offset_key in kernel_buffer_velocity.keys():
            kernels = kernel_buffer_velocity[offset_key]
        else:
            offsets = (offsets.clone().unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)-offset_summary)
            offsets = offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,(orders[0]+1),(orders[1]+1),1,1).detach().requires_grad_(True)
            
            kernels = toCuda(torch.zeros(1,1+2+4+2,(orders[0]+1),(orders[1]+1),2,2))
            for l in range(orders[0]+1):
                for m in range(orders[1]+1):
                    kernels[0:1,0:1,l,m,:,:] = p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
            
            # velocity
            kernels[0:1,1:3,:,:,:,:] = rot(kernels[:,0:1,:,:,:,:],offsets,create_graph=True,retain_graph=True)
            # gradients of velocity
            kernels[0:1,3:5] = grad(kernels[0:1,1:2,:,:,:,:],offsets,create_graph=True,retain_graph=True)
            kernels[0:1,5:7] = grad(kernels[0:1,2:3,:,:,:,:],offsets,create_graph=True,retain_graph=True)
            # laplace of velocity
            kernels[0:1,7:8] = div(kernels[0:1,3:5],offsets,retain_graph=True)
            kernels[0:1,8:9] = div(kernels[0:1,5:7],offsets,retain_graph=False)
            
            kernels = kernels.reshape(1,1+2+4+2,(orders[0]+1)*(orders[1]+1),2,2).detach()
            
            # buffer kernels
            kernel_buffer_velocity[offset_key] = kernels
            save_buffers()

        output = F.conv2d(weights,kernels[0],padding=0)

        # CODO: to be even more efficient, we could separate interpolation in x/y direction
        return output[:,0:1],output[:,1:3],output[:,3:7],output[:,7:9]
        
    

