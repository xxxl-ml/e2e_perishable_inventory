from config.config_loader import load_config, args_add
from utils.data_loader import CustomDataset
from trainers.pto_trainer import PTO_predict
from trainers.ete_trainer import ETE_decision
import torch
import json
import warnings
import gzip
import pickle
import os
warnings.filterwarnings("ignore")

def load_basic_data(args, existing_normalizer=None):
    
    with gzip.open(args['data_save_path'] + f"basic_data_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['data_seed']}.pt", "rb") as f:
        basic_data = pickle.load(f)
    dataset = CustomDataset(basic_data, args, existing_normalizer=existing_normalizer)

    return dataset


def load_path_data(args, dataset):

    # with gzip.open(args['data_save_path'] + f"path_data_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.pt", "rb") as f:
    with gzip.open(args['data_save_path'] + f"path_data_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['data_seed']}.pt", "rb") as f:
        path_data = pickle.load(f)
    dataset.add_path_data(path_data)

    return dataset


def pto_train(dataset, args):

    best_params_D = json.load(open(args['config_path'] + "PTO_structure_best_params_D.json", "r"))
    best_params_L = json.load(open(args['config_path'] + "PTO_structure_best_params_L.json", "r"))
    args_D = args.copy()
    args_D.update(best_params_D)
    args_L = args.copy()
    args_L.update(best_params_L)
    pto_model_D = PTO_predict(args_D, label="D")
    pto_model_L = PTO_predict(args_L, label="L")
    _, train_time_D = pto_model_D.train(dataset)
    _, train_time_L = pto_model_L.train(dataset)

    return pto_model_D, pto_model_L, train_time_D, train_time_L


def ete_train(dataset, args, train_ratio=1.0):

    best_str = json.load(open(args['config_path'] + "ETE_structure_best_params.json", "r"))
    best_lmbda = json.load(open(args['config_path'] + f"{args['model_name']}_lmbda_best_params_seed{args['train_seed']}.json", "r"))
    args_model = args.copy()
    args_model.update(best_str)
    args_model.update(best_lmbda)
    ete_model = ETE_decision(args_model)
    _, train_time = ete_model.train(dataset, train_ratio=train_ratio)

    return ete_model, train_time


if __name__ == "__main__":

    train_time_lst = []

    args = load_config()
    args.update(json.load(open("config/syn_data_corr_random_corr/basic_arg.json", "r")))
    args = args_add(args)

    # args['device'] = "cpu"
    args['phase'] = "train"
    dataset = load_basic_data(args)

    # train the PTO models
    args['model_name'] = "PTO"
    args['early_stop'] = True
    args['delta'] = 0.00005
    args['patience'] = 5
    pto_model_D, pto_model_L, train_time_D, train_time_L = pto_train(dataset, args)
    train_time_lst.append(["PTO_D", train_time_D])
    train_time_lst.append(["PTO_L", train_time_L])

    # train the ETE models
    dataset = load_path_data(args, dataset)
    model_name_lst = ["ETE_order_DLreg", "ETE_target_PILreg_noinstock"]
    args['early_stop'] = True
    args['delta'] = 0.0001
    args['patience'] = 3
    args['phase'] = "train"
    for model_name in model_name_lst:
        args['model_name'] = model_name
        print("model_name: ", model_name)
        ete_model, train_time = ete_train(dataset, args)     
        train_time_lst.append([model_name, train_time])
        torch.cuda.empty_cache()

    # # save the model and corresponding training time into csv file
    # train_time_df = pd.DataFrame(train_time_lst, columns=["model", "train_time"])
    # if not os.path.exists(args['test_save_path']):
    #     os.makedirs(args['test_save_path'])
    # train_time_df.to_csv(args['test_save_path'] + f"train_time.csv", index=False)