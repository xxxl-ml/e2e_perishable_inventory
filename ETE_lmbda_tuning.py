from config.config_loader import load_config, args_add
from trainers.ete_trainer import ETE_decision
from train import load_basic_data, load_path_data
import torch
import pandas as pd
import json
import warnings
import os
warnings.filterwarnings("ignore")


def ETE_tuning(dataset, reg_name, reg_range, args, val_method="cross"):  # cross validation or direct validation

    struc_args = args.copy()
    best_params = json.load(open(args['config_path'] + "ETE_structure_best_params.json", "r"))
    struc_args.update(best_params)
    if val_method == "direct":
        struc_args['early_stop'] = True

    loss_list = []
    best_reg1, best_reg2, best_val_loss = None, None, float("inf")
    best_model, best_training_time = None, None
    for reg1 in reg_range[0]:
        for reg2 in reg_range[1]:
            struc_args[reg_name[0]] = reg1
            struc_args[reg_name[1]] = reg2
            ete_model = ETE_decision(struc_args)
            if val_method == "cross":
                avg_val_loss = ete_model.cross_validation(dataset, k=5)
            elif val_method == "direct":
                avg_val_loss, training_time = ete_model.train(dataset)
            loss_list.append([reg1, reg2, avg_val_loss])
            if avg_val_loss < best_val_loss:
                best_reg1, best_reg2, best_val_loss = reg1, reg2, avg_val_loss
                best_model = ete_model
                if val_method == "direct": best_training_time = training_time
            torch.cuda.empty_cache()
            print(f"{reg_name[0]}: {reg1}, {reg_name[1]}: {reg2}, avg_val_loss: {avg_val_loss}")

    # save the results: including the TC and the lmd_pil0, lmd_pil1
    results = pd.DataFrame(loss_list, columns=[reg_name[0], reg_name[1], "avg_val_loss"])
    if not os.path.exists(args['tuning_save_path']):
        os.makedirs(args['tuning_save_path'])
    results.to_csv(args['tuning_save_path'] + args['model_name'] + f"_lmbda_tuning_seed{args['train_seed']}.csv", index=False)
    # save the best parameter as a json file
    best_params = {reg_name[0]: best_reg1, reg_name[1]: best_reg2}
    if not os.path.exists(args['config_path']):
        os.makedirs(args['config_path'])
    with open(args['config_path'] + args['model_name'] + f"_lmbda_best_params_seed{args['train_seed']}.json", "w") as f:
        json.dump(best_params, f)
    # save the best model
    if val_method == "direct":
        if not os.path.exists(args['model_save_path']):
            os.makedirs(args['model_save_path'])
        torch.save(best_model.model.state_dict(), args['model_save_path'] + f"{args['model_name']}_model_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.pt")

    return loss_list, best_reg1, best_reg2, best_model, best_training_time


if __name__ == "__main__":

    args = load_config()
    args.update(json.load(open("config/syn_data_corr_random_corr/basic_arg.json", "r")))
    args = args_add(args)
    args['save_step'] = args['epochs'] + 1  # no log saving during tuning
    args['phase'] = "train"
    dataset = load_basic_data(args)
    dataset = load_path_data(args, dataset)

    model_lst = ["ETE_order_DLreg", "ETE_target_PILreg_noinstock"]
    for model_name in model_lst:
        args['model_name'] = model_name
        print(f"Start for {args['model_name']}")
        if "DLreg" in model_name:
            reg_name = ["lmd_d", "lmd_l"]
            reg_range = [[0.1, 1.0, 2.5, 5.0], [0.01, 0.1, 0.5, 1.0]]
        elif "PILreg" in model_name:
            reg_name = ["lmd_pil0", "lmd_pil1"]
            reg_range = [[0.1, 1.0, 2.5, 5.0], [0.1, 0.5, 1.0, 2.5]]
        loss_list, best_reg1, best_reg2, best_model, best_training_time = ETE_tuning(dataset, reg_name, reg_range, args, val_method="direct")
        print(f"model_name: {args['model_name']}, best_{reg_name[0]}: {best_reg1}, best_{reg_name[1]}: {best_reg2}")
        torch.cuda.empty_cache()