import torch


class Dataset:


    def __init__(self, width=200, height=200, device=torch.device("cpu")):

        self.width = width
        self.height = height
        self.device = device

        self.h = torch.zeros(self.height, self.width, device=self.device)
        self.u = torch.zeros(self.height, self.width, device=self.device)
        self.v = torch.zeros(self.height, self.width, device=self.device)
        self.s = torch.zeros(self.height, self.width, device=self.device)


    def get(self):
        return h, u, v, s

    def put(self, h_new, u_new, v_new, s_new):
        self.h = h_new
        self.u = u_new
        self.v = v_new
        self.s = s_new
