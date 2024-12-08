import torch
import math
import numpy as np

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.multi_objective.pareto import is_non_dominated

import pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination

from pymoo.core.individual import Individual
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.util.ref_dirs import get_reference_directions
from sklearn.preprocessing import MinMaxScaler

from botorch.utils.sampling import draw_sobol_samples

raw_samples = 512

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def optimize_st_list(acq_func_list, bounds):
	# for qnparego and qucb
	
	candidates, _ = optimize_acqf_list(
		acq_function_list=acq_func_list,
		bounds=bounds,
		num_restarts=2,
		raw_samples=raw_samples,
		options={"batch_limit": 5, "maxiter": 200},
		)

	return candidates

def optimize_st_acqf(acq_func, batch_size, std_bounds):
	# for st qnehvi and qucb
	
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=std_bounds,
        q=batch_size,
        num_restarts=2,
        raw_samples=raw_samples,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    return candidates


def optimize_st_egbo(acq_func, x, y, batch_size):
	# for st qnehvi
	
	pareto_mask = is_non_dominated(y)
	pareto_x = x[pareto_mask].cpu().numpy()
	
	problem = PymooProblem(n_var=x.shape[1], n_obj=y.shape[1], n_constr=0,
                           xl=np.zeros(x.shape[1]), xu=np.ones(x.shape[1]))
    ref_dirs = get_reference_directions("energy", y.shape[1], batch_size*n_tasks)
    algo = NSGA3(pop_size=raw_samples, ref_dirs=ref_dirs)
    
    algo.setup(problem, termination = NoTermination())
    pop = Population.new("X", x.cpu().numpy())
    pop.set("F", -y.cpu().numpy())
    algo.tell(infills=pop)
    new_pop = algo.ask()

	candidates = torch.tensor(new_pop.get("X"), **tkwargs)
		
	acq_value_list = [acq_func(candidates[i].unsqueeze(dim=0)).detach().item()
					  for i in range(candidates.shape[0])]
	sorted_x = candidates.cpu().numpy()[np.argsort(acq_value_list)]
	
	return torch.tensor(sorted_x[-batch_size:], **tkwargs) # take best BATCH_SIZE samples