from config.config_loader import load_config, args_add
from train import load_basic_data, load_path_data
from utils.data_loader import CustomDataset
from trainers.ete_trainer import ETE_decision
import optuna
import torch
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")



def main(args, val_method="cross"):  # cross validation or direct validation

    dataset = load_basic_data(args)
    dataset = load_path_data(args, dataset)
    if val_method == "direct":
        args['early_stop'] = True
    ete_model = ETE_decision(args)
    if val_method == "cross":
        avg_val_loss = ete_model.cross_validation(dataset, k=5)
    elif val_method == "direct":
        avg_val_loss, _ = ete_model.train(dataset)
    else:
        raise ValueError("val_method should be either cross or direct")

    torch.cuda.empty_cache()
    return avg_val_loss


def objective(args, trial):
    
        args_copy = args.copy()
        args_copy["hidden_size_ete"] = param_space["hidden_size_ete"](trial)
        args_copy["embedding_size"] = param_space["embedding_size"](trial)
        args_copy["num_layers"] = param_space["num_layers"](trial)
        args_copy["epochs"] = param_space["epochs"](trial)
        args_copy["batch_size"] = param_space["batch_size"](trial)
        args_copy["weight_decay"] = param_space["weight_decay"](trial)
        args_copy["learning_rate"] = param_space["learning_rate"](trial)
        args_copy["lr_decay"] = param_space["lr_decay"](trial)
        args_copy["decay_step"] = param_space["decay_step"](trial)
    
        avg_val_loss = main(args_copy, val_method="cross")
        return avg_val_loss

if __name__ == "__main__":

    args = load_config()
    args.update(json.load(open("config/syn_data_corr_random_corr/basic_arg.json", "r")))
    args = args_add(args)
    args['save_step'] = args['epochs'] + 1  # no log saving during tuning
    args['model_name'] = "ETE_target_PILreg_noinstock"
    args['phase'] = "train"

    param_space = {
        "hidden_size_ete": lambda trial: trial.suggest_categorical("hidden_size_ete", [[h1, h2, h3, h4] for h1 in [32, 64, 128] for h2 in [32, 64, 128] for h3 in [128, 256] for h4 in [128, 256]]),
        "embedding_size": lambda trial: trial.suggest_categorical("embedding_size", [0, 1, 5, 10, 15, 20]),
        "num_layers": lambda trial: trial.suggest_categorical("num_layers", [1, 2, 3]),
        "epochs": lambda trial: trial.suggest_categorical("epochs", [20, 30, 40]),
        "batch_size": lambda trial: trial.suggest_categorical("batch_size", [64, 128, 256]),
        "weight_decay": lambda trial: trial.suggest_categorical("weight_decay", [0, 1e-6, 1e-5, 1e-4]),
        "learning_rate": lambda trial: trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2]),
        "lr_decay": lambda trial: trial.suggest_categorical("lr_decay", [0.4, 0.6, 0.8]),
        "decay_step": lambda trial: trial.suggest_categorical("decay_step", [1, 3, 5])
    }

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(args, trial), n_trials=50)
    # save all the results
    results = study.trials_dataframe()
    results.to_csv(args['tuning_save_path'] + "ETE_structure_tuning.csv", index=False)
    # save the best parameter as a json file
    best_params = study.best_params
    with open(args['config_path'] + "ETE_structure_best_params.json", "w") as f:
        json.dump(best_params, f)