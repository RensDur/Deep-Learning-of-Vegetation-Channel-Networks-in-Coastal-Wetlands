import torch
from torch import nn
import numpy as np
from unet_parts import *

def get_Net(params, spline_variables):
	if params.net == "Fluid_model":
		net = fluid_model(orders_v=[params.orders_v,params.orders_v],orders_p=[params.orders_p,params.orders_p],hidden_size=params.hidden_size,input_size=3)
	elif params.net == "Wave_model":
		params.orders_p = params.orders_v = params.orders_z
		net = wave_model(orders_v=[params.orders_v,params.orders_v],orders_p=[params.orders_p,params.orders_p],hidden_size=params.hidden_size,input_size=2,residuals=True)
	elif params.net == "ShallowWaterModelCNN":
		net = ShallowWaterModel(spline_variables, hidden_size=params.hidden_size)
	elif params.net == "ShallowWaterModelUNet":
		net = ShallowWaterUNet(spline_variables, hidden_size=params.hidden_size)
	elif params.net == "ShallowWaterUNet + SedimentUNet":
		net = (
			ShallowWaterUNet(spline_variables, hidden_size=params.hidden_size),
			SedimentUNet(spline_variables, hidden_size=params.hidden_size)
		)
	elif params.net == "ShallowWaterUNet + SedimentUNet + VegetationUNet":
		net = (
			ShallowWaterUNet(spline_variables, hidden_size=params.hidden_size),
			SedimentUNet(spline_variables, hidden_size=params.hidden_size),
			VegetationUNet(spline_variables, hidden_size=params.hidden_size)
		)
	return net




class ShallowWaterUNet(nn.Module):
	# inspired by UNet taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
	
	def __init__(self, spline_variables, hidden_size=64, interpolation_size=8, bilinear=True, input_size=4, residuals=False):
		"""
		:orders_v: order of spline for velocity potential (should be at least 2)
		:orders_p: order of spline for pressure field
		:hidden_size: hidden size of neural net
		:interpolation_size: size of first interpolation layer for v_cond and v_mask
		"""
		super(ShallowWaterUNet, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear
		self.input_size = input_size
		
		self.spline_variables = spline_variables
		self.hidden_input_size = spline_variables.hidden_size()
		self.hidden_output_size = spline_variables["h"].hidden_size() + spline_variables["u"].hidden_size() + spline_variables["v"].hidden_size()
		self.residuals = residuals
		
		self.interpol = nn.Conv2d(input_size,interpolation_size,kernel_size=2) # interpolate v_cond (2) and v_mask (1) from 4 surrounding fields
		self.inc = DoubleConv(self.hidden_input_size+interpolation_size, hidden_size) # input: hidden_state + interpolation of v_cond and v_mask
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, self.hidden_output_size)

		self.output_scalar = torch.ones(1, self.hidden_output_size, 1, 1)*2

		self.output_scalar[:,spline_variables.get_singular_slice_for("h"),:,:] = 2
		self.output_scalar[:,spline_variables.get_singular_slice_for("u"),:,:] = 10
		self.output_scalar[:,spline_variables.get_singular_slice_for("v"),:,:] = 10

	def to(self, torch_device):
		super(ShallowWaterUNet, self).to(torch_device)
		self.output_scalar = self.output_scalar.to(torch_device)
		return self
	
	def forward(self, hidden_state, closed_mask, opened_mask, h_mask, h_cond):
		"""
		:hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		:v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
		:v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
		:return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		"""
		x = torch.cat([closed_mask, opened_mask, h_mask, h_cond],dim=1)
		
		x = self.interpol(x)
		
		x = torch.cat([hidden_state,x],dim=1)
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
		out = x
		
		# residual connections
		extracted_water_hidden_state = torch.cat([
			self.spline_variables.extract_from(hidden_state, "h"),
			self.spline_variables.extract_from(hidden_state, "u"),
			self.spline_variables.extract_from(hidden_state, "v")
		], dim=1)
		out[:,:,:,:] = self.output_scalar*torch.tanh((out[:,:,:,:]+extracted_water_hidden_state[:,:,:,:])/self.output_scalar)
		
		# # Substract the mean of every variable
		# out[:,self.spline_variables.get_singular_slice_for("h"),:,:] = out[:,self.spline_variables.get_singular_slice_for("h"),:,:] - torch.mean(out[:,self.spline_variables.get_singular_slice_for("h"),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		# out[:,self.spline_variables.get_singular_slice_for("hu"),:,:] = out[:,self.spline_variables.get_singular_slice_for("hu"),:,:] - torch.mean(out[:,self.spline_variables.get_singular_slice_for("hu"),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		# out[:,self.spline_variables.get_singular_slice_for("hv"),:,:] = out[:,self.spline_variables.get_singular_slice_for("hv"),:,:] - torch.mean(out[:,self.spline_variables.get_singular_slice_for("hv"),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)

		return out


class SedimentUNet(nn.Module):
	# inspired by UNet taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
	
	def __init__(self, spline_variables, hidden_size=64, interpolation_size=4, bilinear=True, input_size=2, residuals=False):
		"""
		:orders_v: order of spline for velocity potential (should be at least 2)
		:orders_p: order of spline for pressure field
		:hidden_size: hidden size of neural net
		:interpolation_size: size of first interpolation layer for v_cond and v_mask
		"""
		super(SedimentUNet, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear
		self.input_size = input_size
		
		self.spline_variables = spline_variables
		self.hidden_input_size = spline_variables.hidden_size()
		self.hidden_output_size = spline_variables["s"].hidden_size()
		self.residuals = residuals
		
		self.interpol = nn.Conv2d(input_size,interpolation_size,kernel_size=2) # interpolate v_cond (2) and v_mask (1) from 4 surrounding fields
		self.inc = DoubleConv(self.hidden_input_size+interpolation_size, hidden_size) # input: hidden_state + interpolation of v_cond and v_mask
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, self.hidden_output_size)

		self.output_scalar = torch.ones(1, self.hidden_output_size, 1, 1)*2

	def to(self, torch_device):
		super(SedimentUNet, self).to(torch_device)
		self.output_scalar = self.output_scalar.to(torch_device)
		return self
	
	def forward(self, hidden_state, closed_mask, opened_mask):
		"""
		:hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		:v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
		:v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
		:return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		"""
		x = torch.cat([closed_mask, opened_mask],dim=1)
		
		x = self.interpol(x)
		
		x = torch.cat([hidden_state,x],dim=1)
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
		out = x
		
		# residual connections
		extracted_sediment_hidden_state = self.spline_variables.extract_from(hidden_state, "s")
		out[:,:,:,:] = self.output_scalar*torch.tanh((out[:,:,:,:]+extracted_sediment_hidden_state[:,:,:,:])/self.output_scalar)
		
		# # Substract the mean of every variable
		# out[:,self.spline_variables.get_singular_slice_for("h"),:,:] = out[:,self.spline_variables.get_singular_slice_for("h"),:,:] - torch.mean(out[:,self.spline_variables.get_singular_slice_for("h"),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		# out[:,self.spline_variables.get_singular_slice_for("hu"),:,:] = out[:,self.spline_variables.get_singular_slice_for("hu"),:,:] - torch.mean(out[:,self.spline_variables.get_singular_slice_for("hu"),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		# out[:,self.spline_variables.get_singular_slice_for("hv"),:,:] = out[:,self.spline_variables.get_singular_slice_for("hv"),:,:] - torch.mean(out[:,self.spline_variables.get_singular_slice_for("hv"),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)

		return out


class VegetationUNet(nn.Module):
	# inspired by UNet taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
	
	def __init__(self, spline_variables, hidden_size=64, interpolation_size=4, bilinear=True, input_size=2, residuals=False):
		"""
		:orders_v: order of spline for velocity potential (should be at least 2)
		:orders_p: order of spline for pressure field
		:hidden_size: hidden size of neural net
		:interpolation_size: size of first interpolation layer for v_cond and v_mask
		"""
		super(VegetationUNet, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear
		self.input_size = input_size
		
		self.spline_variables = spline_variables
		self.hidden_input_size = spline_variables.hidden_size()
		self.hidden_output_size = spline_variables["b"].hidden_size()
		self.residuals = residuals
		
		self.interpol = nn.Conv2d(input_size,interpolation_size,kernel_size=2) # interpolate v_cond (2) and v_mask (1) from 4 surrounding fields
		self.inc = DoubleConv(self.hidden_input_size+interpolation_size, hidden_size) # input: hidden_state + interpolation of v_cond and v_mask
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, self.hidden_output_size)

		self.output_scalar = torch.ones(1, self.hidden_output_size, 1, 1)*1500

	def to(self, torch_device):
		super(VegetationUNet, self).to(torch_device)
		self.output_scalar = self.output_scalar.to(torch_device)
		return self
	
	def forward(self, hidden_state, closed_mask, opened_mask):
		"""
		:hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		:v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
		:v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
		:return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		"""
		x = torch.cat([closed_mask, opened_mask],dim=1)
		
		x = self.interpol(x)
		
		x = torch.cat([hidden_state,x],dim=1)
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
		out = x
		
		# residual connections
		extracted_sediment_hidden_state = self.spline_variables.extract_from(hidden_state, "b")
		out[:,:,:,:] = self.output_scalar*torch.tanh((out[:,:,:,:]+extracted_sediment_hidden_state[:,:,:,:])/self.output_scalar)
		
		# # Substract the mean of every variable
		# out[:,self.spline_variables.get_singular_slice_for("h"),:,:] = out[:,self.spline_variables.get_singular_slice_for("h"),:,:] - torch.mean(out[:,self.spline_variables.get_singular_slice_for("h"),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		# out[:,self.spline_variables.get_singular_slice_for("hu"),:,:] = out[:,self.spline_variables.get_singular_slice_for("hu"),:,:] - torch.mean(out[:,self.spline_variables.get_singular_slice_for("hu"),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		# out[:,self.spline_variables.get_singular_slice_for("hv"),:,:] = out[:,self.spline_variables.get_singular_slice_for("hv"),:,:] - torch.mean(out[:,self.spline_variables.get_singular_slice_for("hv"),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)

		return out


class fluid_model(nn.Module):
	# inspired by UNet taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
	
	def __init__(self, orders_v,orders_p,hidden_size=64,interpolation_size=5, bilinear=True,input_size=3,residuals=False):
		"""
		:orders_v: order of spline for velocity potential (should be at least 2)
		:orders_p: order of spline for pressure field
		:hidden_size: hidden size of neural net
		:interpolation_size: size of first interpolation layer for v_cond and v_mask
		"""
		super(fluid_model, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear
		self.input_size = input_size
		
		self.orders_v = orders_v
		self.orders_p = orders_p
		self.v_size = np.prod([i+1 for i in orders_v])
		self.p_size = np.prod([i+1 for i in orders_p])
		self.hidden_state_size = self.v_size + self.p_size
		self.residuals = residuals
		
		self.interpol = nn.Conv2d(input_size,interpolation_size,kernel_size=2) # interpolate v_cond (2) and v_mask (1) from 4 surrounding fields
		self.inc = DoubleConv(self.hidden_state_size+interpolation_size, hidden_size) # input: hidden_state + interpolation of v_cond and v_mask
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, self.hidden_state_size)
		self.output_scalar = toCuda(torch.ones(1,self.v_size+self.p_size,1,1)*2)
		self.output_scalar[:,0:1,:,:] = 400
		self.output_scalar[:,(self.v_size):(self.v_size+1),:,:] = 400
		
	
	def forward(self,hidden_state,v_cond,v_mask):
		"""
		:hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		:v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
		:v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
		:return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		"""
		x = torch.cat([v_cond,v_mask],dim=1)
		
		x = self.interpol(x)
		
		x = torch.cat([hidden_state,x],dim=1)
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
		out = x
		
		# residual connections
		out[:,:,:,:] = self.output_scalar*torch.tanh((out[:,:,:,:]+hidden_state[:,:,:,:])/self.output_scalar)
		
		#substract mean of a_z and p
		out[:,0:1,:,:] = out[:,0:1,:,:]-torch.mean(out[:,0:1,:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		out[:,(self.v_size):(self.v_size+1),:,:] = out[:,(self.v_size):(self.v_size+1),:,:]-torch.mean(out[:,(self.v_size):(self.v_size+1),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		return out

class wave_model(nn.Module):
	
	def __init__(self, orders_v,orders_p,hidden_size=64,interpolation_size=5, bilinear=True,input_size=3,residuals=False):
		"""
		:orders_v: order of spline for velocity potential (should be at least 2)
		:orders_p: order of spline for pressure field
		:hidden_size: hidden size of neural net
		:interpolation_size: size of first interpolation layer for v_cond and v_mask
		"""
		super(wave_model, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear
		self.input_size = input_size
		
		self.orders_v = orders_v
		self.orders_p = orders_p
		self.v_size = np.prod([i+1 for i in orders_v])
		self.p_size = np.prod([i+1 for i in orders_p])
		self.hidden_state_size = self.v_size + self.p_size
		self.residuals = residuals
		
		self.interpol = nn.Conv2d(input_size,interpolation_size,kernel_size=2) # interpolate v_cond (2) and v_mask (1) from 4 surrounding fields
		self.conv1 = nn.Conv2d(self.hidden_state_size+interpolation_size, self.hidden_size,kernel_size=3,padding=1) # input: hidden_state + interpolation of v_cond and v_mask
		self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size,kernel_size=3,padding=1) # input: hidden_state + interpolation of v_cond and v_mask
		self.conv3 = nn.Conv2d(self.hidden_size, self.hidden_state_size,kernel_size=3,padding=1) # input: hidden_state + interpolation of v_cond and v_mask
		
		if self.hidden_state_size == 18: # if orders_z = 2
			self.output_scalar_wave = toCuda(torch.Tensor([5,0.5,0.05,0.5, 0.05,0.05,0.05,0.05,0.05, 5,0.5,0.05,0.5, 0.05,0.05,0.05,0.05,0.05]).unsqueeze(0).unsqueeze(2).unsqueeze(3))
		elif self.hidden_state_size == 8: # if orders_z = 1
			self.output_scalar_wave = toCuda(torch.Tensor([5,0.5,0.5,0.05, 5,0.5,0.5,0.05]).unsqueeze(0).unsqueeze(2).unsqueeze(3))
		
	
	def forward(self,hidden_state,v_cond,v_mask):
		"""
		:hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		:v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
		:v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
		:return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		"""
		x = torch.cat([v_cond,v_mask],dim=1)
		
		x = self.interpol(x)
		
		x = torch.cat([hidden_state,x],dim=1)
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		out = self.conv3(x)
		
		# residual connections
		out[:,:,:,:] = self.output_scalar_wave*torch.tanh((out[:,:,:,:]+hidden_state[:,:,:,:]/self.output_scalar_wave))
		
		return out