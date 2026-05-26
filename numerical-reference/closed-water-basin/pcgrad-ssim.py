import torch
import torch.nn.functional as F
from window import MultiWindow
from ssim import *
from spline.spline_variable import SplineVariable
from spline.spline_array import SplineArray
from main import ClosedWaterBasin


class SplineInterpolator:

    def __init__(self, width, height, device=torch.device("cpu")):

        # Dimensions
        self.width = width
        self.height = height

        # Torch device
        self.device = device

        # Variables in this dataset
        self.variables = SplineArray(
            SplineVariable("h", 1, requires_derivative=True, requires_laplacian=True),
            SplineVariable("u", 2, requires_derivative=True, requires_laplacian=True),
            SplineVariable("v", 2, requires_derivative=True, requires_laplacian=True),
            device=self.device
        )


    def interpolate_states(self, hidden_state, offset):
        """
        :old_hidden_states: old hidden states (size: bs x (v_size+p_size) x w x h)
        :new_hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
        :offset: offset in x / y / t direction (vector of size 3 containing values between 0 and 1)
        :return: interpolated fields for:
            :z: z field
            :grad(z): gradient of z field
            :laplace(z): laplacian of z field
            :dz/dt: velocity of z field
            :dz^2/dt^2: acceleration of z field
        """

        # z field: requires first derivative
        h, grad_h, _ = self.variables["h"].interpolate_at(self.variables.extract_from(hidden_state, "h"), offset)

        # u field: requires first derivative + laplace
        u, grad_u, _ = self.variables["u"].interpolate_at(self.variables.extract_from(hidden_state, "u"), offset)

        # v field: requires first derivative + laplace
        v, grad_v, _ = self.variables["v"].interpolate_at(self.variables.extract_from(hidden_state, "v"), offset)

        return h, u, v

    def interpolate_superres(self, hidden_states, resolution_factor):
        """
        :hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
        "resolution_factor": resolution factor for superres interpolation
        :return: interpolated fields for:
            :z: z field
            :grad(z): gradient of z field
            :laplace(z): laplacian of z field
            :dz/dt: velocity of z field
            :dz^2/dt^2: acceleration of z field
        """

        # h field: requires first derivative
        h, grad_h, _ = self.variables["h"].interpolate_superres_at(self.variables.extract_from(hidden_states, "h"), resolution_factor)

        # u field: requires first derivative + laplace
        u, grad_u, _ = self.variables["u"].interpolate_superres_at(self.variables.extract_from(hidden_states, "u"), resolution_factor)

        # v field: requires first derivative + laplace
        v, grad_v, _ = self.variables["v"].interpolate_superres_at(self.variables.extract_from(hidden_states, "v"), resolution_factor)

        return h, grad_h, u, grad_u, v, grad_v


if __name__ == "__main__":
    
    window = MultiWindow(200, 200)

    basin = ClosedWaterBasin(200, 200, torch.device("cpu"))

    # Load the PINN simulation data from disk
    pinn_solution = torch.load(f"./swe-std-output/reflection.pt")

    interpolator = SplineInterpolator(200, 200, torch.device("cpu"))

    simulation_count = 0

    ssim_scores = []
    ssim_plot, = window.axs[1, 2].plot([], [])

    # Open the window
    window.open()

    # As long as the window is open, run the simulation
    while window.is_open:

        # Make a simulation step
        basin.simulate()

        # Interpolate pinn solution
        pinn_image, _, _, _, _, _ = interpolator.interpolate_superres(pinn_solution[simulation_count:(simulation_count+1)], 1)
        pinn_image = pinn_image.to(torch.device("cpu"))

        # Calculate the ssim score
        ssim_spatial = ssim(basin.h, pinn_image)
        ssim_scores.append(torch.mean(ssim_spatial).cpu().item())

        # Update the window state
        window.set_data(
            basin.h[0, 0],
            basin.u[0, 0],
            basin.v[0, 0],
            pinn_image[0, 0],
            ssim_spatial[0, 0]
        )

        simulation_count += 1

        ssim_plot.set_xdata(range(len(ssim_scores)))
        ssim_plot.set_ydata(ssim_scores)
        window.axs[1, 2].relim()
        window.axs[1, 2].autoscale_view()

        window.update()