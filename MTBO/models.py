import torch
import gpytorch

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_model_st(x, y):

    models = []
    for i in range(y.shape[-1]):
        models.append(SingleTaskGP(x, y[..., i : i + 1],
                                   outcome_transform=Standardize(m=1)))

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    return model, mll