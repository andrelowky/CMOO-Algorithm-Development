import torch
import numpy as np
import math

#import joblib
#import os

#surrogate = joblib.load(f'{os.getcwd()}/MTBO/surrogate')
#yscaler = joblib.load(f'{os.getcwd()}/MTBO/yscaler')

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

class DTLZ1():
    def __init__(self, n_var = 8, delta2 = 0, negate=True):
        self.n_var = n_var
        self.n_obj = 3
        self.negate = negate

        bounds = torch.zeros((2, n_var), **tkwargs)
        bounds[1] = 1
        self.bounds = bounds

        if self.negate:
           self.ref_pt = torch.tensor([-400, -400, -400], **tkwargs) 
        else:
            self.ref_pt = torch.tensor([400, 400, 400], **tkwargs)
                
        self.delta1 = 1
        self.delta2 = delta2
        self.delta3 = 1

        self.hv_scaling = 100000
    
    def evaluate(self, x):

        M = 3
        g = 100*self.delta3*(8 + torch.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2 - torch.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)), axis=1 )) + (1-self.delta3)*(-20*torch.exp(-0.2*torch.sqrt(torch.mean( ((x[:,M-1:] -0.5 - self.delta2)*50)**2, axis=1 ) )) - torch.exp(torch.mean(torch.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)*50) , axis=1 ) ) + 20 + np.e)

        f1 = 0.5*self.delta1*self.delta3*x[:,0]*x[:,1]*(1+g) + (1-self.delta3)*(1+g)*torch.cos(x[:,0]*np.pi/2)*torch.cos(x[:,1]*np.pi/2)
        f2 = 0.5*self.delta1*self.delta3*x[:,0]*(1-x[:,1])*(1+g) + (1-self.delta3)*(1+g)*torch.cos(x[:,0]*np.pi/2)*torch.sin(x[:,1]*np.pi/2)
        f3 = 0.5*self.delta1*self.delta3*(1-x[:,0])*(1+g) + (1-self.delta3)*(1+g)*torch.sin(x[:,0]*np.pi/2)

        if self.negate:
            return -1* torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)])
        else:
            return torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)])


class DTLZ2():
    def __init__(self, n_var = 8, delta2 = 0, negate=True):
        self.n_var = n_var
        self.n_obj = 3
        self.negate = negate

        bounds = torch.zeros((2, n_var), **tkwargs)
        bounds[1] = 1
        self.bounds = bounds

        if self.negate:
           self.ref_pt = torch.tensor([-400, -400, -400], **tkwargs) 
        else:
            self.ref_pt = torch.tensor([400, 400, 400], **tkwargs)
                
        self.delta1 = 1
        self.delta2 = delta2
        self.delta3 = 1

        self.hv_scaling = 100000

    def evaluate(self, x):

        M = 3

        g = 100*torch.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2, axis=1 ) + (1-self.delta3)*(-20*torch.exp(-0.2*torch.sqrt( torch.mean( ((x[:,M-1:] - 0.5 - self.delta2)*50)**2, axis=1 ) )) - torch.exp( torch.mean( torch.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)*50) , axis=1 ) ) + 20 + np.e)

        f1 = (1+g)*self.delta1*self.delta3*torch.cos(x[:,0]*np.pi/2)*torch.cos(x[:,1]*np.pi/2) + 0.5*(1 - self.delta3)*x[:,0]*x[:,1]*(1+g)
        f2 = (1+g)*self.delta1*self.delta3*torch.cos(x[:,0]*np.pi/2)*torch.sin(x[:,1]*np.pi/2) + 0.5*(1 - self.delta3)*x[:,0]*(1-x[:,1])*(1+g)
        f3 = (1+g)*self.delta1*self.delta3*torch.sin(x[:,0]*np.pi/2) + 0.5*(1 - self.delta3)*(1-x[:,0])*(1+g)

        if self.negate:
            return -1* torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)])
        else:
            return torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)])

'''
class asc16():
	def __init__(self, delta1):
		self.n_var = 5
		self.n_obj = 2
		self.delta1 = delta1
	
		bounds = torch.zeros((2, self.n_var), **tkwargs)
		bounds[1] = 1
		self.bounds = bounds
	
		self.ref_pt = torch.tensor([-1.1, 0], **tkwargs) 
		self.hv_scaling = 1
	
	def evaluate(self, x):
		x_np = x.cpu().numpy()
		y = surrogate.predict(x_np)
		y = yscaler.inverse_transform(y.reshape(1,-1))
	
		f1 = -x[:,1].unsqueeze(1) #temperature, to be minimized
		f2 = torch.tensor(y, **tkwargs).reshape(-1, 1)
		f2 = f2 + self.delta1
		
		return torch.hstack([f1, f2])
'''

class ZDT1():
	def __init__(self, n_var = 8, delta1 = 0, delta2 = 0, negate=True):
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
				
		self.delta1 = delta1
		self.delta2 = delta2
	
		self.hv_scaling = 1
	
	def evaluate(self, x):
	
		f1 = x[:, 0]
		g = 1 + (torch.sin(4*np.pi*x[:, 0])*self.delta1) + 9.0 / (self.n_var - 1) * torch.sum(x[:, 1:], axis=1)
		f2 = (g * (1 - torch.pow((f1 / g), 0.5)) ) + (torch.sum(x[:, 1:], axis=1) * self.delta2)
	
		if self.negate:
			return -1* torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1)])
		else:
			return torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1)])

class ZDT2():
	def __init__(self, n_var = 8, delta1 = 0, delta2 = 0, negate=True):
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
				
		self.delta1 = delta1
		self.delta2 = delta2
	
		self.hv_scaling = 1
	
	def evaluate(self, x):
	
		f1 = x[:, 0]
		c = torch.sum(x[:, 1:], axis=1)
		g = 1.0 + (4*np.pi*torch.sin(x[:, 0])*self.delta1) + 9.0 * c / (self.n_var - 1)
		f2 = ( g * (1 - torch.pow((f1 * 1.0 / g), 2)) ) + (torch.sum(x[:, 1:], axis=1) * self.delta2)
	
		if self.negate:
			return -1* torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1)])
		else:
			return torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1)])