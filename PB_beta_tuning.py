from config.config_loader import load_config, args_add
from trainers.pto_trainer import PTO_predict
from env.simulation import simulator
from train import load_basic_data
import torch
import pandas as pd
import numpy as np
import json
import warnings
import os
warnings.filterwarnings("ignore")


def load_pto_model(args, dataset, best_params_D, best_params_L):

    args_D = args.copy()
    args_D.update(best_params_D)
    pto_model_D = PTO_predict(args_D, label="D")
    pto_model_D.train(dataset, existing_model=torch.load(args['model_save_path'] + f"D_model_seed{args['train_seed']}.pt"))
    args_L = args.copy()
    args_L.update(best_params_L)
    pto_model_L = PTO_predict(args_L, label="L")
    pto_model_L.train(dataset, existing_model=torch.load(args['model_save_path'] + f"L_model_seed{args['train_seed']}.pt"))

    return pto_model_D, pto_model_L


def PB_tuning(args, beta_range, dataset=None):
    
    if dataset is None:
        dataset = load_basic_data(args)
    best_params_D = json.load(open(args['config_path'] + "PTO_structure_best_params_D.json", "r"))
    best_params_L = json.load(open(args['config_path'] + "PTO_structure_best_params_L.json", "r"))
    pto_model_D, pto_model_L = load_pto_model(args, dataset, best_params_D, best_params_L)

    loss_list = []
    best_beta, best_loss = None, float("inf")
    for beta_adjust in beta_range:
        args_copy = args.copy()
        args_copy["beta_adjust"] = beta_adjust
        simu = simulator(args_copy, initial_stock_path=False)
        TC, HC, BC, OC, SO, OD, time_cost =  simu.simulate(dataset, pto_model=[pto_model_D, pto_model_L])
        print(f"beta_adjust: {beta_adjust}, TC: {TC}, HC: {HC}, BC: {BC}, OC: {OC}, SO: {SO}, OD:{OD}, time_cost: {time_cost}")
        torch.cuda.empty_cache()
        loss_list.append([beta_adjust, TC])
        if len(loss_list) >= 2 and loss_list[-1][1] > loss_list[-2][1]:  # the loss is increasing, break the loop
            break
        if TC < best_loss:
            best_loss = TC
            best_beta = beta_adjust

    # save the results: including the TC and the beta_adjust
    results = pd.DataFrame(loss_list, columns=["beta_adjust", "TC"])
    if not os.path.exists(args['tuning_save_path']):
        os.makedirs(args['tuning_save_path'])
    results.to_csv(args['tuning_save_path'] + f"PB_tuning_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.csv", index=False)

    # save the best parameter as a json file
    best_params = {"beta_adjust": best_beta}
    if not os.path.exists(args['config_path']):
        os.makedirs(args['config_path'])
    with open(args['config_path'] + f"PB_best_params_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.json", "w") as f:
        json.dump(best_params, f)

    return loss_list, best_beta


if __name__ == "__main__":

    args = load_config()   
    args.update(json.load(open("config/syn_data_corr_random_corr/basic_arg.json", "r")))
    args = args_add(args)
    args['save_step'] = args['epochs'] + 1

    args['model_name'] = "PB_tuning"
    args['phase'] = "train"

    beta_range = np.linspace(-0.5, 0.5, 21)
    loss_list, best_beta = PB_tuning(args, beta_range)