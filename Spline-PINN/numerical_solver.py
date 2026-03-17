import torch
import torch.nn.functional as F





class NumericalSolver:


    def __init__(self, params, device):

        #
        # Store local copy of the parameters
        #
        self.params = params

        #
        # Torch device
        #
        self.device = device

    

    def step(self, h, grad_h, hu, grad_hu, hv, grad_hv, s, grad_s, laplacian_s, dt, h_in, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask):

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
        # COMPUTE NUMERICAL STEP
        #

        # h step
        dh_dt = - grad_hu[:,1:2] - grad_hv[:,0:1] + h_in

        # hu step
        dhu_dt = - self.params.grav*h*(grad_s[:,1:2] + grad_h[:,1:2]) - hu*(du_dx + dv_dy) - u*grad_hu[:,1:2] - v*grad_hu[:,0:1] - tau_bx_per_rho

        # hv step
        dhv_dt = - self.params.grav*h*(grad_s[:,0:1] + grad_h[:,0:1]) - hv*(du_dx + dv_dy) - u*grad_hv[:,1:2] - v*grad_hv[:,0:1] - tau_by_per_rho

        # s step
        ds_dt = s_switch * (self.params.Sin * (he / (self.params.Qs + he)) - self.params.Es * s * tau_b_per_rho + self.params.D0 * laplacian_s)

        # Update quantities
        h = h + dh_dt * dt
        hu = hu + dhu_dt * dt
        hv = hv + dhv_dt * dt
        s = s + ds_dt * dt

        # Boundary conditions
        h[torch.where(h_mask > 0)] = (h_cond + self.params.H0)[torch.where(h_mask > 0)]

        hu[torch.where(uv_mask > 0)] = (uv_cond)[torch.where(uv_mask > 0)]
        hv[torch.where(uv_mask > 0)] = (uv_cond)[torch.where(uv_mask > 0)]

        s[torch.where(s_mask > 0)] = (s_cond)[torch.where(s_mask > 0)]

        return dh_dt, dhu_dt, dhv_dt, ds_dt