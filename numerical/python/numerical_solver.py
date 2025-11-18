import torch
from domain import Domain

class NumericalSolver:


    def __init__(self):
        self.device = torch.device('cpu') # Default to CPU
        # Switch to MPS (Apple Metal) if available
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        # Or CUDA if we're on an Nvidia machine
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')

        print(f"Using torch device '{self.device}'")

        # Domain
        self.domain = Domain(100, 100, 0.5)
        self.domain.init_preset_gaussian_wave(25, 25, 5, 5)
        self.domain.to(self.device)

        # Timestep
        self.dt = 0.0001

    def solve_step(self):
        
        grid_copy = self.domain.grid.clone()

        for x in range(self.domain.width):
            for y in range(self.domain.height):
                left = grid_copy[y, max(0, x-1)]
                right = grid_copy[y, min(self.domain.width-1, x+1)]
                top = grid_copy[max(0, y-1), x]
                bottom = grid_copy[min(self.domain.height-1, y+1), x]

                # Computing change in water layer thickness h
                duh_dx = 

                
