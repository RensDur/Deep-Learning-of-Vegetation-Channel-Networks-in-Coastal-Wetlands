import torch
from spline.spline_variable import SplineVariable


torch_device = torch.device("mps")

order = 2

test = SplineVariable("test", order, requires_derivative=True, requires_laplacian=False, device=torch_device)

hidden_size = (order+1)*(order+1)

weights = torch.zeros(1, hidden_size, 10, 1)

offsets = [torch.rand(3)]

resolution_factor = 4
j = 0

offset = torch.floor(offsets[j]*resolution_factor)/resolution_factor
offset = offset.to(torch_device)

# output = test.interpolate_at(weights, offset[:2])

# print(output)