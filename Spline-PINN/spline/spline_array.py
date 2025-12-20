from spline.spline_variable import SplineVariable
import torch
import numpy as np

class SplineArray:

	def __init__(self, *variables: list[SplineVariable], device=torch.device("cpu")):
		
		# SplineVariables are stored in an array in a fixed order
		self.variables: list[SplineVariable] = variables

		# Store the torch device for later reference
		self.device = device

		# Update the device of each variable inserted
		for v in self.variables:
			v.to(self.device)

	def append(self, variable: SplineVariable):
		self.variables.append(variable)

	def get_slice_for(self, varname):
		start = 0

		for v in self.variables:
			if varname == v.get_name():
				return slice(start, start + v.hidden_size())
			
			start += v.hidden_size()

		return None
	
	def extract_from(self, hidden_states, varname):
		return hidden_states[self.get_slice_for(varname)]

	def __len__(self):
		return len(self.variables)
	
	def hidden_size(self):
		return np.sum([v.hidden_size() for v in self.variables])

	def __getitem__(self, items):
		if type(items) == int:
			return self.variables[items]
		elif type(items) == str:
			for v in self.variables:
				if items == v.get_name():
					return v
			raise Exception(f"No SplineVariable with name {items}")
		raise Exception(f"Unsupported indexing method for SplineArray: {type(items)}")