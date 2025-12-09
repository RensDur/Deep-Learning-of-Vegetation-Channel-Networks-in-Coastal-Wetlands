import torch
from torch import nn
import torch.nn.functional as F
from unet_parts import *

def get_Net(params):
	if params.net == "UNet1":
		pde_cnn = PDE_UNet1(params.hidden_size)
	elif params.net == "UNet2":
		pde_cnn = PDE_UNet2(params.hidden_size)
	elif params.net == "UNet3":
		pde_cnn = PDE_UNet3(params.hidden_size)
	elif params.net == "UNetSWE":
		pde_cnn = PDE_CNN_SWE(params.hidden_size)
	return pde_cnn

class PDE_UNet_SWE(nn.Module):
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet_SWE, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(3, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 3)

	def forward(self, h_old, u_old, v_old):
		x = torch.cat([h_old, u_old, v_old],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)

		h_delta = 10 * torch.tanh((x[:,0:1]) / 10)

		# Normalize delta in h to make sure that the total amount of water in the basin always remains
		h_delta = h_delta.data-torch.mean(h_delta.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
		h_new = h_old + h_delta

		u_new = 400 * torch.tanh((u_old + x[:,1:2]) / 400)
		v_new = 400 * torch.tanh((v_old + x[:,2:3]) / 400)

		return h_new, u_new, v_new
	
class PDE_CNN_SWE(nn.Module):
	def __init__(self, hidden_size=32,bilinear=True):
		super(PDE_CNN_SWE, self).__init__()
		self.hidden_size = 50
		self.bilinear = bilinear

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3,padding=1, padding_mode='replicate')
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3,padding=1, padding_mode='replicate')
		self.conv3 = nn.Conv2d(64, 32, kernel_size=3,padding=1, padding_mode='replicate')
		self.conv4 = nn.Conv2d(32, 3, kernel_size=3,padding=1, padding_mode='replicate')

	def forward(self, h_old, u_old, v_old):
		x = torch.cat([h_old, u_old, v_old],dim=1)
		
		x = self.conv1(x)
		x = torch.relu(x)
		x = self.conv2(x)
		x = torch.relu(x)
		x = self.conv3(x)
		x = torch.relu(x)
		x = self.conv4(x)

		h_new = 10 * torch.tanh((h_old + x[:,0:1]) / 10)
		u_new = 50 * torch.tanh((u_old + x[:,1:2]) / 50)
		v_new = 50 * torch.tanh((v_old + x[:,2:3]) / 50)

		return h_new, u_new, v_new


class PDE_UNet1(nn.Module):
	#inspired by UNet taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
	
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet1, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(13, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 3)

	def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
		v_old = rot_mac(a_old)
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		a_new, p_new = 400*torch.tanh(x[:,0:1]/400), 10*torch.tanh(x[:,1:2]/10)
		return a_new,p_new

class PDE_UNet2(nn.Module):
	#same as UNet1 but with delta a / delta p
	
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet2, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(13, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 2)

	def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
		v_old = rot_mac(a_old)
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		a_new, p_new = 400*torch.tanh((a_old+x[:,0:1])/400), 10*torch.tanh((p_old+x[:,1:2])/10)
		return a_new,p_new


class PDE_UNet3(nn.Module):
	#same as UNet2 but with scaling
	
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet3, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(13, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 4)

	def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
		v_old = rot_mac(a_old)
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		a_new, p_new = 400*torch.tanh((a_old+x[:,0:1]*torch.exp(3*torch.tanh(x[:,2:3]/3)))/400), 10*torch.tanh((p_old+x[:,1:2]*torch.exp(3*torch.tanh(x[:,3:4]/3)))/10)
		return a_new,p_new

