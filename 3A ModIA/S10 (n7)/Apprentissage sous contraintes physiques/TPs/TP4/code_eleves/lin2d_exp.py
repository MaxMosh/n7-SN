"""Save experiment dict exemple
"""
import torch
import manage_exp


list_k_update = [{}]

def name_fun(k_update):
    """Generates an experiment name
    """
    return "lin2d_exp"

x_dim = 2
b_size = 2**7 # mb
m = 2 # ensemble size
h_dim = x_dim*m
activation_classname = "nn.LeakyReLU"

sigma0 = 0.01 # std of x0
sigmap = sigma0 # std of prop
sigmao = 10*sigma0 # std of obs

# Default dict
k = {}
# - experiment
k["tensor_type"] = "double" # "float"
k["seed"] = 0
# -- net
k["net_classname"] = "filters.DAN"
k["net_kwargs"] = {
    "a_kwargs": {
        "loc_classname": "FcZeroLin",
        "loc_kwargs": {
            "in_dim": h_dim + x_dim,
            "out_dim": h_dim,
            "deep": 1,
            "activation_classname": activation_classname}},
    "b_kwargs": {
        "loc_classname": "FcZero",
        "loc_kwargs": {
            "dim": h_dim,
            "deep": 1,
            "activation_classname": activation_classname}},
    "c_kwargs": {
        "loc_classname": "FullyConnected",
        "loc_kwargs": {
            "layers": [h_dim, x_dim+x_dim*(x_dim+1)//2], 
            "activation_classname": activation_classname},
        "gauss_dim": x_dim}}
k["sigma0"] = sigma0
k["prop_kwargs"] = {
    "loc_classname": "Lin2d",
    "loc_kwargs": {
        "x_dim": x_dim,
        "N": 1,
        "dt": 0,
        "init": "0"},
    "gauss_dim": x_dim,
    "scale_vec_classname": "Cst",
    "scale_vec_kwargs": {"init": torch.log(torch.tensor([sigmap]))}}
k["obs_kwargs"] = {
    "loc_classname": "Id",
    "loc_kwargs": {},
    "gauss_dim": x_dim,
    "scale_vec_classname": "Cst",
    "scale_vec_kwargs": {"init": torch.log(torch.tensor([sigmao]))}}
# -- train
k["train_kwargs"] = {
    "mode": 'full',
    "b_size": b_size,
    "h_dim": h_dim,
    "x_dim": x_dim,
    "T": 50,
    "checkpoint": 10}
k["test_kwargs"] = {
    "b_size": b_size,
    "h_dim": h_dim,
    "x_dim": x_dim,
    "T": 3,
    "checkpoint": 1}
k["optimizer_classname"] = "torch.optim.LBFGS"
k["optimizer_kwargs"] = {"max_iter": 1000, "max_eval": 2000, "line_search_fn": 'strong_wolfe',\
                         "tolerance_grad": 1e-14, "tolerance_change": 1e-14,\
                         "history_size": 20
                        }
k["scheduler_classname"] = "NONE"
k["scheduler_kwargs"] = {}
k["directory"] = "./"

def get_params():
    return k
    
if __name__ == "__main__":
    manage_exp.update_and_save(k, list_k_update, name_fun)
