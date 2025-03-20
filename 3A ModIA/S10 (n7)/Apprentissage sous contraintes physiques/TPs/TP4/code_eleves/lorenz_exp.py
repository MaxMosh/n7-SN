"""Save experiment dict exemple
"""
import torch
import manage_exp


list_k_update = [{}]

def name_fun(k_update):
    """Generates an experiment name
    """
    return "lorenz_exp"

x_dim = 40
b_size = 2**10
h_dim = x_dim*20
activation_classname = "nn.LeakyReLU"

sigma0 = 1.0 # std of x0
sigmap = 0.1*sigma0 # std of prop
sigmao = sigma0 # std of obs

# Default dict
k = {}
# - experiment
k["tensor_type"] = "float"
k["seed"] = 100
# -- net
k["net_classname"] = "filters.DAN"
k["net_kwargs"] = {
    "a_kwargs": {
        "loc_classname": "FcZeroLin",
        "loc_kwargs": {
            "in_dim": h_dim + x_dim,
            "out_dim": h_dim,
            "deep": 10,
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
            "layers": [h_dim, x_dim*(x_dim+1)//2],
            "activation_classname": activation_classname},
        "gauss_dim": x_dim}}
k["sigma0"] = sigma0
k["prop_kwargs"] = {
    "loc_classname": "EDO",
    "loc_kwargs": {
        "x_dim": x_dim,
        "N": 1,
        "dt": .05,
        "init": "95"},
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
    "mode": 'online',
    "b_size": b_size,
    "h_dim": h_dim,
    "x_dim": x_dim,
    "T": 10000,
    "checkpoint": 100}
k["test_kwargs"] = {
    "b_size": b_size,
    "h_dim": h_dim,
    "x_dim": x_dim,
    "T": 1000,
    "checkpoint": 100}
# -- optimizer
k["optimizer_classname"] = "torch.optim.Adam"
k["optimizer_kwargs"] = {"lr": 10**-4}
k["scheduler_classname"] = "torch.optim.lr_scheduler.ExponentialLR"
k["scheduler_kwargs"] = {"gamma": 1}
k["directory"] = "./"

def get_params():
    return k

if __name__ == "__main__":
    manage_exp.update_and_save(k, list_k_update, name_fun)

