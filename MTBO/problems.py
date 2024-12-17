import torch
import numpy as np
import math

#import joblib
#import os

#surrogate = joblib.load(f'{os.getcwd()}/MTBO/surrogate')
#yscaler = joblib.load(f'{os.getcwd()}/MTBO/yscaler')

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

class ZDT1():
	def __init__(self, n_var = 8, negate=True):
		self.n_var = n_var
		self.n_obj = 2
		self.negate = negate
	
		bounds = torch.zeros((2, n_var), **tkwargs)
		bounds[1] = 1
		self.bounds = bounds
	
		if self.negate:
		   self.ref_pt = torch.tensor([-11, -11], **tkwargs) 
		else:
			self.ref_pt = torch.tensor([11, 1], **tkwargs)
	
		self.hv_scaling = 1
	
	def evaluate(self, x):
	
		f1 = x[:, 0]
		g = 1 + 9.0 / (self.n_var - 1) * torch.sum(x[:, 1:], axis=1)
		f2 = g * (1 - torch.pow((f1 / g), 0.5))
	
		if self.negate:
			return -1* torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1)])
		else:
			return torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1)])

class ZDT2():
	def __init__(self, n_var = 8, negate=True):
		self.n_var = n_var
		self.n_obj = 2
		self.negate = negate
	
		bounds = torch.zeros((2, n_var), **tkwargs)
		bounds[1] = 1
		self.bounds = bounds
	
		if self.negate:
		   self.ref_pt = torch.tensor([-11, -11], **tkwargs) 
		else:
			self.ref_pt = torch.tensor([11, 11], **tkwargs)
	
		self.hv_scaling = 1
	
	def evaluate(self, x):
	
		f1 = x[:, 0]
		c = torch.sum(x[:, 1:], axis=1)
		g = 1.0 + 9.0 * c / (self.n_var - 1)
		f2 = g * (1 - torch.pow((f1 * 1.0 / g), 2))
	
		if self.negate:
			return -1* torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1)])
		else:
			return torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1)])
