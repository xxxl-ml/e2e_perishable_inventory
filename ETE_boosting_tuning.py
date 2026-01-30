from config.config_loader import load_config, args_add
from trainers.ete_trainer import ETE_decision
from env.simulation import simulator
from train import load_basic_data, load_path_data
import torch
import pandas as pd
import numpy as np
import json
import warnings
import os
warnings.filterwarnings("ignore")


def load_ete_model(args, dataset):

    best_str = json.load(open(args['config_path'] + "ETE_structure_best_params.json", "r"))
    # best_lmbda = json.load(open(args['config_path'] + f"{args['model_name'][:-len("_boosting_tuning")]}_lmbda_best_params_seed{args['train_seed']}.json", "r"))
    args_model = args.copy()
    args_model.update(best_str)
    # args_model.update(best_lmbda)
    ete_model = ETE_decision(args_model)
    ete_model.train(dataset, existing_model=torch.load(args['model_save_path'] + f"{args['model_name'][:-len('_boosting_tuning')]}_model_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.pt"))

    return ete_model

def ETE_boosting_tuning(args, gamma_range, dataset=None):
    
    if dataset is None:
        dataset = load_basic_data(args)
        dataset = load_path_data(args, dataset)
    ete_model = load_ete_model(args, dataset)

    loss_list = []
    best_gamma, best_loss = None, float("inf")
    for boosting_gamma in gamma_range:
        args_copy = args.copy()
        args_copy["boosting_gamma"] = boosting_gamma
        simu = simulator(args_copy, initial_stock_path=False)
        TC, HC, BC, OC, SO, OD, time_cost =  simu.simulate(dataset, ete_model=ete_model)
        print(f"boosting_gamma: {boosting_gamma}, TC: {TC}, HC: {HC}, BC: {BC}, OC: {OC}, SO: {SO}, OD:{OD}, time_cost: {time_cost}")
        torch.cuda.empty_cache()
        loss_list.append([boosting_gamma, TC])
        if len(loss_list) >= 2 and loss_list[-1][1] > loss_list[-2][1]:  # the loss is increasing, break the loop
            break
        if TC < best_loss:
            best_loss = TC
            best_gamma = boosting_gamma

    # save the results: including the TC and the boosting_gamma
    results = pd.DataFrame(loss_list, columns=["boosting_gamma", "TC"])
    if not os.path.exists(args['tuning_save_path']):
        os.makedirs(args['tuning_save_path'])
    results.to_csv(args['tuning_save_path'] + f"{args['model_name']}_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.csv", index=False)

    # save the best parameter as a json file
    best_params = {"boosting_gamma": best_gamma}
    if not os.path.exists(args['config_path']):
        os.makedirs(args['config_path'])
    with open(args['config_path'] + f"{args['model_name'][:-len('_tuning')]}_best_gamma_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.json", "w") as f:
        json.dump(best_params, f)

    return loss_list, best_gamma


if __name__ == "__main__":

    args = load_config()   
    args.update(json.load(open("config/syn_data_corr_random_corr/basic_arg.json", "r")))
    args = args_add(args)
    args['save_step'] = args['epochs'] + 1

    args['model_name'] = "ETE_target_PILreg_noinstock_boosting_tuning"
    args['phase'] = "train"

    gamma_range = np.linspace(0.5, 1.5, 21)
    loss_list, best_gamma = ETE_boosting_tuning(args, gamma_range)