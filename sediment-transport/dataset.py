import torch


class Dataset:


    def __init__(self, width=200, height=200, device=torch.device("cpu")):

        self.width = width
        self.height = height
        self.device = device

        self.h = torch.zeros(1, 1, self.height, self.width, device=self.device)
        self.hu = torch.zeros(1, 1, self.height, self.width, device=self.device)
        self.hv = torch.zeros(1, 1, self.height, self.width, device=self.device)
        self.s = torch.zeros(1, 1, self.height, self.width, device=self.device)

    def start_condition(self, name):

        if name == "rest-lake":
            self.h[:, :, :, :] = 1
            self.hu[:, :, :, :] = 0.1
            self.hu[:, :, (self.height//2-10):(self.height//2+10), self.width//2] = 0
            self.hu[:, :, (self.height//2-10):(self.height//2+10), self.width//2+1] = 0

        if name == "wave-left-right":
            self.h[:, :, :, :] = 1
            self.h[:, :, :, 0:5] = 1.5

        if name == "center-square":
            self.h[:, :, :, :] = 1
            self.h[:, :, (self.height//2-5):(self.height//2+5), (self.width//2-5):(self.width//2+5)] = 1.5

    def get(self):
        return self.h, self.hu, self.hv, self.s

    def put(self, h_new, u_new, v_new, s_new):
        self.h = h_new
        self.hu = u_new
        self.hv = v_new
        self.s = s_new
