import torch
import torch.nn.functional as F
from dataset import Dataset



class LossCalculator:


    def __init__(self, dataset: Dataset, params, device):

        #
        # Dataset
        #
        self.dataset = dataset

        #
        # Store local copy of the parameters
        #
        self.params = params

        #
        # Torch device
        #
        self.device = device

        #
        # Diffusion operation (needed, if we want to put more loss-weight to regions close to the domain boundaries)
        #
        self.kernel_width = 3
        self.kernel = torch.exp(-torch.arange(-2,2.001,4/(2*self.kernel_width)).float()**2)
        self.kernel /= torch.sum(self.kernel)
        self.kernel_x = self.kernel.unsqueeze(0).unsqueeze(1).unsqueeze(3).to(self.device)
        self.kernel_y = self.kernel.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(self.device)

    def diffuse(self, T):
        """
        Needed to put extra weight on domain borders
        """
        T = F.conv2d(T,self.kernel_x,padding=[self.kernel_width,0])
        T = F.conv2d(T,self.kernel_y,padding=[0,self.kernel_width])
        return T

    def loss_function(self, x):
        # return F.huber_loss(x, torch.zeros_like(x), reduction="none", delta=self.params.huber_delta)
        return x**2

    def compute_batch_loss(self, old_hidden_state, new_hidden_state, grid_offsets, h_in, sample_h_conds, sample_h_masks, sample_uv_conds, sample_uv_masks, sample_s_conds, sample_s_masks, dim=[1,2,3]):

        # Compute Physics Informed Loss image tensor
        loss_h = 0
        loss_u = 0
        loss_v = 0
        loss_s = 0
        loss_bound = 0
        loss_damp = 0

        # Go over each sample
        for j, sample in enumerate(grid_offsets):
            offset = torch.floor(sample*self.params.resolution_factor)/self.params.resolution_factor

            # For added clarity: The masks define where the BCs act, they're 1 everywhere on the boundary, 0 everywhere else
            sample_h_cond = sample_h_conds[j]
            sample_h_mask = sample_h_masks[j]
            sample_uv_cond = sample_uv_conds[j]
            sample_uv_mask = sample_uv_masks[j]
            sample_s_cond = sample_s_conds[j]
            sample_s_mask = sample_s_masks[j]

            sample_h_domain_mask = 1-sample_h_mask
            sample_uv_domain_mask = 1-sample_uv_mask
            sample_s_domain_mask = 1-sample_s_mask

            # Put additional border_weight on domain boundaries:
            # Important: weighed by parameter 'border_weight'
            sample_h_mask = (sample_h_mask + sample_h_mask*self.diffuse(sample_h_domain_mask)*self.params.border_weight).detach()
            sample_uv_mask = (sample_uv_mask + sample_uv_mask*self.diffuse(sample_uv_domain_mask)*self.params.border_weight).detach()
            sample_s_mask = (sample_s_mask + sample_s_mask*self.diffuse(sample_s_domain_mask)*self.params.border_weight).detach()

            # Interpolate spline coefficients to obtain the necessary quantities
            h, grad_h, dh_dt, hu, grad_hu, dhu_dt, hv, grad_hv, dhv_dt, s, grad_s, laplacian_s, ds_dt = self.dataset.interpolate_states(old_hidden_state, new_hidden_state, offset)

            # Add mean water level height
            h = h + self.params.H0
            h = F.relu(h - self.params.Hc) + self.params.Hc

            #
            # Derive u and v
            #
            u = hu / h
            v = hv / h

            #
            # Derive grad(u) and grad(v) via the quotient rule
            #
            du_dx = (1.0 / torch.pow(h, 2)) * (h * grad_hu[:,1:2] - hu * grad_h[:,1:2])
            du_dy = (1.0 / torch.pow(h, 2)) * (h * grad_hu[:,0:1] - hu * grad_h[:,0:1])

            dv_dx = (1.0 / torch.pow(h, 2)) * (h * grad_hv[:,1:2] - hv * grad_h[:,1:2])
            dv_dy = (1.0 / torch.pow(h, 2)) * (h * grad_hv[:,0:1] - hv * grad_h[:,0:1])

            #
            # Derive bed friction coefficients
            #

            # n: Manning's coefficient
            n = self.params.nb # + (self.params.nv - self.params.nb) * B / self.params.k

            # Cz: Chezy coefficient
            chezy = (1.0 / n) * torch.pow(h, 1.0 / 6.0)

            # Bed friction components
            # Add really small value to u2+v2 to prevent dividing by zero in backprop (deriv of sqroot is 1/sqrt)
            tau_precalc = (self.params.grav / torch.pow(chezy, 2)) * torch.pow(torch.pow(u, 2) + torch.pow(v, 2) + 1e-12, 0.5)
            tau_bx_per_rho = tau_precalc * u
            tau_by_per_rho = tau_precalc * v
            tau_b_per_rho  = (self.params.grav / torch.pow(chezy, 2)) * (torch.pow(u, 2) + torch.pow(v, 2))

            # Effective water height
            he = h - self.params.Hc

            # Create a mask to capture where s should be compared to the PDE or zero
            # This can be done based on Hin, which will be zero for hydrodynamic environments
            s_switch = h_in * (1.0 / self.params.Hin)

            #
            # COMPUTE SAMPLE LOSS
            #

            # h-loss
            loss_h = loss_h + torch.mean(self.loss_function(
                dh_dt + grad_hu[:,1:2] + grad_hv[:,0:1] - h_in[:,:,1:-1,1:-1]
            ), dim)

            # Momentum loss
            loss_u = loss_u + torch.mean(self.loss_function(
                dhu_dt + self.params.grav*h*(grad_s[:,1:2] + grad_h[:,1:2]) + hu*(du_dx + dv_dy) + u*grad_hu[:,1:2] + v*grad_hu[:,0:1] + tau_bx_per_rho
            ), dim)

            loss_v = loss_v + torch.mean(self.loss_function(
                dhv_dt + self.params.grav*h*(grad_s[:,0:1] + grad_h[:,0:1]) + hv*(du_dx + dv_dy) + u*grad_hv[:,1:2] + v*grad_hv[:,0:1] + tau_by_per_rho
            ), dim)

            # Sediment loss
            loss_s = loss_s + torch.mean(self.loss_function(
                ds_dt - s_switch[:,:,1:-1,1:-1] * (self.params.Sin * (he / (self.params.Qs + he)) + self.params.Es * s * tau_b_per_rho - self.params.D0 * laplacian_s)
            ), dim)

            # h boundary condition loss
            loss_bound_h = torch.mean(sample_h_mask[:,:,1:-1,1:-1] * self.loss_function(
                h - (sample_h_cond[:,:,1:-1,1:-1] + self.params.H0)
            ), dim)

            # Boundary condition loss
            loss_bound_uv_grad_h = torch.mean(sample_uv_mask[:,:,1:-1,1:-1] * self.loss_function(
                grad_h
            ), dim)

            loss_bound_uv_grad_s = torch.mean(sample_uv_mask[:,:,1:-1,1:-1] * self.loss_function(
                grad_s
            ), dim)

            loss_bound_u = torch.mean(sample_uv_mask[:,:,1:-1,1:-1] * self.loss_function(
                hu - sample_uv_cond[:,:,1:-1,1:-1]
            ), dim)

            loss_bound_v = torch.mean(sample_uv_mask[:,:,1:-1,1:-1] * self.loss_function(
                hv - sample_uv_cond[:,:,1:-1,1:-1]
            ), dim)

            # Sedimentary BC loss
            loss_bound_s = torch.mean(sample_s_mask[:,:,1:-1,1:-1] * self.loss_function(
                s - sample_s_cond[:,:,1:-1,1:-1]
            ), dim)

            loss_bound_s_grad_hu = torch.mean(sample_s_mask[:,:,1:-1,1:-1] * self.loss_function(
                grad_hu
            ), dim)

            loss_bound_s_grad_hv = torch.mean(sample_s_mask[:,:,1:-1,1:-1] * self.loss_function(
                grad_hv
            ), dim)

            loss_bound = loss_bound + loss_bound_h + loss_bound_uv_grad_h + loss_bound_uv_grad_s + loss_bound_u + loss_bound_v + loss_bound_s + loss_bound_s_grad_hu + loss_bound_s_grad_hv

            # Damping loss
            # loss_damp_h = torch.mean(self.loss_function(grad_h), dim)
            # loss_damp_u = torch.mean(self.loss_function(u), dim)
            # loss_damp_v = torch.mean(self.loss_function(v), dim)

            # loss_damp = loss_damp + self.damp_loss_factor * (loss_damp_h + loss_damp_u + loss_damp_v)

        # Multiply by the loss weights
        loss_h = loss_h * self.params.loss_h
        loss_u = loss_u * self.params.loss_momentum
        loss_v = loss_v * self.params.loss_momentum
        loss_s = loss_s * self.params.loss_s * self.params.morphological_acc_factor
        loss_bound = loss_bound * self.params.loss_bound

        # Normalize towards the number of samples taken
        loss_h = loss_h / self.params.n_samples
        loss_u = loss_u / self.params.n_samples
        loss_v = loss_v / self.params.n_samples
        loss_s = loss_s / self.params.n_samples
        loss_bound = loss_bound / self.params.n_samples
        # loss_damp = loss_damp / self.params.n_samples

        return loss_h, loss_u, loss_v, loss_s, loss_bound