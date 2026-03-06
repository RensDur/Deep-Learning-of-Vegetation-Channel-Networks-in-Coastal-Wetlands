


class DatasetBoundaryCondition:

    def __init__(self, params, device=torch.device("cpu")):

        # Local copy of the parameters
        self.params = params

        # Store the torch device
        self.device = device

        # Store the low-res mask & condition
        self.mask = torch.zeros(self.dataset_size, 1, self.width, self.height)
        self.cond = torch.zeros(self.dataset_size, 1, self.width, self.height)

        # Store the high-res mask & condition
        self.mask_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)
        self.cond_fullres = torch.zeros(self.dataset_size, 1, self.width_fullres, self.height_fullres)

        