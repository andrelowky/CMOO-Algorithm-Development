import torch
import numpy as np
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.transforms import unnormalize, normalize, standardize

from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def calc_hv(y, task, hv, problems, negate=True, hv_scale=10000):
    volume_list = []
    for i, problem in enumerate(problems):
        specific_y = y[(task == i).all(dim=1)]
            
        pareto_mask = is_non_dominated(specific_y)
        pareto_y = specific_y[pareto_mask]
        volume_list.append(hv.compute(pareto_y)/problems[0].hv_scaling)

    return np.array(volume_list)

def update_values(old_tensors, new_tensors, problems):
	old_x, old_task, old_y = old_tensors
	new_x, new_task = new_tensors
	
	for i, problem in enumerate(problems):
		added_x = new_x[(new_task==i).all(dim=1)]
		if added_x.shape[0] == 0:
			continue;
		added_task = new_task[(new_task==i).all(dim=1)]
		added_y = problem.evaluate(added_x)
	
		old_x = torch.cat([old_x, added_x])
		old_task = torch.cat([old_task, added_task])
		old_y = torch.cat([old_y, added_y])
	
	return old_x, old_task, old_y


def calc_losses(model, mll):
    with torch.no_grad():
        outputs = model(*model.train_inputs)
        loss = -mll(outputs, model.train_targets).item()

    return abs(loss)


def trust_threshold(x, task, y, idx, threshold=0.5):
	n_task = task.max().item()+1
	
	x_std = standardize(x).cpu().numpy()
	y_std = []
	for i in range(int(n_task)):
		y_i = y[(task==i).all(dim=1)]
		y_std.append(standardize(y_i).cpu().numpy())

	y_std = np.array(y_std).reshape(y.shape)
		
	x_np = x.cpu().numpy()
	y_np = y.cpu().numpy()
	
	base_mask = (task==idx).all(dim=1).cpu().numpy()
	other_mask = (task!=idx).all(dim=1).cpu().numpy()
	
	xy_std = np.hstack([x_std, y_std])
	xy_pca = PCA(n_components=1).fit_transform(x_std)
	kde = gaussian_kde(xy_pca[base_mask].T)
	prob_dens = kde(xy_pca[other_mask].T)
	trust_threshold = (prob_dens>=(prob_dens*threshold)) # even if none left, thats okay
	
	x_all = torch.vstack([x[base_mask], x[other_mask][trust_threshold]])
	y_all = torch.vstack([y[base_mask], y[other_mask][trust_threshold]])	

	return x_all, y_all


