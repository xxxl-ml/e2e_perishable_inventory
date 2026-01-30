from config.config_loader import load_config, args_add
from trainers.pto_trainer import PTO_predict
from trainers.ete_trainer import ETE_decision
from env.simulation import simulator
from train import load_basic_data, load_path_data
import torch
import json
import pandas as pd
import warnings
import os
warnings.filterwarnings("ignore")


def pto_test(dataset, args, saving_frame):

    # load the model
    best_params_D = json.load(open(args['config_path'] + "PTO_structure_best_params_D.json", "r"))
    best_params_L = json.load(open(args['config_path'] + "PTO_structure_best_params_L.json", "r"))
    args_D = args.copy()
    args_D.update(best_params_D)
    args_L = args.copy()
    args_L.update(best_params_L)
    pto_model_D = PTO_predict(args_D, label="D")
    pto_model_L = PTO_predict(args_L, label="L")
    pto_model_D.train(dataset, existing_model=torch.load(args['model_save_path'] + f"D_model_seed{args['train_seed']}.pt"))
    pto_model_L.train(dataset, existing_model=torch.load(args['model_save_path'] + f"L_model_seed{args['train_seed']}.pt"))

    # test the model: including PTO/PSTO/PB/PPB
    simu = simulator(args, initial_stock_path=False)
    TC, HC, BC, OC, SO, OD, time_cost = simu.simulate(dataset, pto_model=[pto_model_D, pto_model_L])

    # save the results
    saving_frame.append([TC, HC, BC, OC, SO, OD, time_cost])

    print(f"model: {args['model_name']}, TC: {TC}, HC: {HC}, BC: {BC}, OC: {OC}, SO: {SO}, OD:{OD}, time_cost: {time_cost}")

    return saving_frame


def pto_test_save(dataset, args):  # for sampling the testing on-hand inventory input

    # load the model
    best_params_D = json.load(open(args['config_path'] + "PTO_structure_best_params_D.json", "r"))
    best_params_L = json.load(open(args['config_path'] + "PTO_structure_best_params_L.json", "r"))
    args_D = args.copy()
    args_D.update(best_params_D)
    args_L = args.copy()
    args_L.update(best_params_L)
    pto_model_D = PTO_predict(args_D, label="D")
    pto_model_L = PTO_predict(args_L, label="L")
    pto_model_D.train(dataset, existing_model=torch.load(args['model_save_path'] + f"D_model_seed{args['train_seed']}.pt"))
    pto_model_L.train(dataset, existing_model=torch.load(args['model_save_path'] + f"L_model_seed{args['train_seed']}.pt"))

    # test the model: including PTO/PSTO/PB/PPB
    simu = simulator(args, initial_stock_path=False)
    TC, HC, BC, OC, SO, OD, time_cost, instock_test, pil_test = simu.simulate(dataset, pto_model=[pto_model_D, pto_model_L], test_save=True)

    return instock_test, pil_test


def ete_test(dataset, args, saving_frame):

    # load the model
    best_str = json.load(open(args['config_path'] + "ETE_structure_best_params.json", "r"))
    args_model = args.copy()
    args_model.update(best_str)
    ete_model = ETE_decision(args_model)
    if "boosting" in args['model_name']:
        best_gamma = json.load(open(args['config_path'] + f"{args['model_name']}_best_gamma_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.json", "r"))
        args.update(best_gamma)
        ete_model.train(dataset, existing_model=torch.load(args['model_save_path'] + f"{args['model_name'][:-len('_boosting')]}_model_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.pt"))
    else:
        args['boosting_gamma'] = 1.0
        ete_model.train(dataset, existing_model=torch.load(args['model_save_path'] + f"{args['model_name']}_model_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.pt"))

    # test the model
    simu = simulator(args, initial_stock_path=False)
    TC, HC, BC, OC, SO, OD, time_cost = simu.simulate(dataset, ete_model=ete_model)
    saving_frame.append([TC, HC, BC, OC, SO, OD, time_cost])
    print(f"model: {args['model_name']}, TC: {TC}, HC: {HC}, BC: {BC}, OC: {OC}, SO: {SO}, OD:{OD}, time_cost: {time_cost}")

    return saving_frame 


def ete_test_save(dataset, args):

    # load the model
    best_str = json.load(open(args['config_path'] + "ETE_structure_best_params.json", "r"))
    args_model = args.copy()
    args_model.update(best_str)
    ete_model = ETE_decision(args_model)
    ete_model.train(dataset, existing_model=torch.load(args['model_save_path'] + f"{args['model_name']}_model_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.pt"))

    # test the model
    simu = simulator(args, initial_stock_path=False)
    TC, HC, BC, OC, SO, OD, time_cost, instock_test, pil_test = simu.simulate(dataset, ete_model=ete_model, test_save=True)

    return instock_test, pil_test


if __name__ == "__main__":

    measures = []

    args = load_config()
    args.update(json.load(open("config/syn_data_corr_random_corr/basic_arg.json", "r")))
    args = args_add(args)
    dataset = load_basic_data(args)

    args['phase'] = "test"
    args['train_seed'] = 0

    args['model_name'] = "PTO_NV"
    measures = pto_test(dataset, args, measures)

    args['model_name'] = "PB"
    args['beta_adjust'] = 0.0
    measures = pto_test(dataset, args, measures)

    args['model_name'] = "PPB"
    best_beta = json.load(open(args['config_path'] + f"PB_best_params_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.json", "r"))
    args.update(best_beta)
    measures = pto_test(dataset, args, measures)

    ete_lst = ["ETE_order_DLreg", "ETE_target_PILreg_noinstock"]
    for model_name in ete_lst:
        args['model_name'] = model_name
        measures = ete_test(dataset, args, measures)
    
    args['model_name'] = "ETE_target_PILreg_noinstock_boosting"
    measures = ete_test(dataset, args, measures)

    results = pd.DataFrame(measures, columns=["TC", "HC", "BC", "OC", "SO", "OD", "evaluation_time"])
    model_lst = ["PTO_NV", "PB", "PPB"] + ete_lst + ["ETE_target_PILreg_noinstock_boosting"]
    results.insert(0, "model_name", model_lst)
    results.to_csv(args['test_save_path'] + f"point_measure_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.csv", index=False)