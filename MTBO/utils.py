import torch
import numpy as np
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.transforms import unnormalize, normalize, standardize


tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def calc_hv(y, task, hv, problem, negate=True):

	pareto_mask = is_non_dominated(y)
	pareto_y = y[pareto_mask]
	volume = hv.compute(pareto_y)/problem.hv_scaling

    return volume
