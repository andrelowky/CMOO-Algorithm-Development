import torch
import math
import time
import numpy as np
import pandas as pd

from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples


#from MTBO.sampling import anchored_sampling
#from MTBO.utils import calc_hv, update_values, calc_losses
from MTBO.models import initialize_model_st
from MTBO.acq_func import st_qnehvi, st_qnparego, st_qucb
from MTBO.optim import optimize_st_list,  optimize_st_acqf, optimize_st_egbo

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}


class MTBO():

	def __init__(self, problem):
		
		self.problem = problem
		self.prob_bounds = problem[0].bounds
		
		self.std_bounds = torch.zeros((2, problem.n_var), **tkwargs)
		self.std_bounds[1] = 1

		self.ref_pt = problem.ref_pt
		self.hv = Hypervolume(ref_point=problems.ref_pt)
		self.n_obj = problem.n_obj
		self.n_var = problem.n_var

		self.results = []

	def initialize(self, n_init, random_state=np.random.randint(99999)):
		# split half for repeated, and half for coverage

		self.random_state = random_state
		self.n_init = n_init
		#self.init_x, self.init_task, self.init_y = anchored_sampling(self.problems, n_init, self.random_state)
	
	def run(self, n_iter, n_batch, 
			task_type, algo,
		   ):
		print(f"Optimizing for {task_type}-{algo}")

		self.did_validation = False
		self.task_type = task_type
		self.run_iter = n_iter
		self.run_batch = n_batch
		self.losses = []

		torch.manual_seed(self.random_state)
		np.random.seed(self.random_state)
		
		#### initialization ####
		
		self.train_x, self.train_y = self.init_x, self.init_y
		self.x_gp = normalize(self.train_x, self.prob_bounds)   

		volumes = calc_hv(self.train_y, self.hv)
		self.results.append(volumes)
		
		print(f"Batch 0 - avg HV:{volumes.mean():.4f}")
		
		for iter in range(1, n_iter+1):
			t2 = time.monotonic()
		
			if task_type == 'single':
				if n_batch < self.n_task:
					raise Exception(f"Batch size too small for fixed sampling! Need at least {self.n_task}.")

				if n_batch % self.n_task != 0:
					raise Exception(f"Batch size should be an even number wrt to {self.n_task} tasks.")

				n_batch_per_task = int(n_batch/self.n_task)

				new_x = []
				all_losses = []
				for i in range(self.n_task):
					
					if algo == 'random':
						model, acq = None, None
						all_losses.append(0)
						new_x_i = draw_sobol_samples(bounds=self.prob_bounds, 
													 n=n_batch_per_task, q=1).squeeze(1)	

						new_x.append(new_x_i.cpu().numpy())

					else:
						x_gp_i = self.x_gp[(self.train_task==i).all(dim=1)]
						train_y_i = self.train_y[(self.train_task==i).all(dim=1)]
		
						model, mll = initialize_model_st(x_gp_i, train_y_i)
						fit_gpytorch_mll(mll)
						all_losses.append(calc_losses(model, mll))
	
						if algo == 'qnehvi':
							acq = st_qnehvi(model, self.ref_pt, x_gp_i)
							candidates = optimize_st_acqf(acq, n_batch_per_task, self.std_bounds)
						elif algo == 'qnehvi-egbo':
							acq = st_qnehvi(model, self.ref_pt, x_gp_i)
							candidates = optimize_st_egbo(acq, x_gp_i, train_y_i, n_batch_per_task)
						elif algo == 'qnparego':
							acq = st_qnparego(model, x_gp_i, n_batch_per_task, self.n_obj)
							candidates = optimize_st_list(acq, self.std_bounds)
						elif algo == 'qucb':
							acq = st_qucb(model, x_gp_i, n_batch_per_task, self.n_obj)
							candidates = optimize_st_list(acq, self.std_bounds)
		
						new_x.append(unnormalize(candidates, self.prob_bounds).cpu().numpy())

				self.losses.append(np.array(all_losses).mean(axis=0))
			
				new_x = torch.tensor(np.array(new_x), **tkwargs).reshape(-1, self.n_var)
				new_task = torch.tensor([np.array(task).repeat(n_batch_per_task) for task in range(self.n_task)], **tkwargs).reshape(-1).unsqueeze(1)

			#### update and go next iteration
			'''
            self.train_x, self.train_task, self.train_y = update_values(
				(self.train_x, self.train_task, self.train_y), 
				(new_x, new_task), self.problems)
            '''
			self.x_gp = normalize(self.train_x, self.prob_bounds)   

			volumes = calc_hv(self.train_y, self.hv)
			self.results.append(volumes)
			
			t3 = time.monotonic()
			print(f"Batch {iter} - avg HV:{volumes.mean():.4f}, time:{t3-t2:>4.2f}")
			
			del model, acq
			torch.cuda.empty_cache()

	def output_results(self):

		hv_results = pd.DataFrame(np.array(self.results), columns=[f"Problem {x+1}" for x in range(self.n_task)])
		
		df_x = pd.DataFrame(self.train_x.cpu().numpy(), columns=[f"x{x}" for x in range(self.n_var)])
		df_y = pd.DataFrame(self.train_y.cpu().numpy(), columns=[f"y{x}" for x in range(self.n_obj)])

        df_iter = pd.DataFrame(np.concatenate(
            [np.repeat(0, self.n_init),
             np.repeat(np.arange(self.run_iter)+1, self.run_batch),]), columns=['Iteration'])
        
		if self.problem.negate:
		    df_y = -df_y

		df_all = pd.concat([df_x, df_y, df_iter], axis=1)

		losses_results = self.losses
		
		return hv_results, df_all, losses_results

	def reset(self):
		self.results = []
		self.train_x = None
		self.train_task = None
		self.train_y = None