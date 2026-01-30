from config.config_loader import load_config, args_add
from utils.data_loader import CustomDataset
from trainers.pto_trainer import PTO_predict
import optuna
import torch
import os
import json
import warnings
import gzip
import pickle
warnings.filterwarnings("ignore")

def main(args, label, val_method='cross', loss='MSE'):  # cross validation or direct validation
    
    with gzip.open(args['data_save_path'] + f'basic_data_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['seed']}.pt', 'rb') as f:
        basic_data = pickle.load(f)
    if val_method == 'direct':
        args['early_stop'] = True
    dataset = CustomDataset(basic_data, args)
    pto_model = PTO_predict(args, label=label, loss=loss)
    if val_method == 'cross':
        avg_val_loss = pto_model.cross_validation(dataset, k=5)
    else:
        avg_val_loss, _ = pto_model.train(dataset)

    torch.cuda.empty_cache()
    return avg_val_loss

def objective(args, trial, label, loss='MSE'):

    args_copy = args.copy()
    args_copy['hidden_size_pto'] = param_space['hidden_size_pto'](trial)
    args_copy['embedding_size'] = param_space['embedding_size'](trial)
    args_copy['num_layers'] = param_space['num_layers'](trial)
    args_copy['epochs'] = param_space['epochs'](trial)
    args_copy['batch_size'] = param_space['batch_size'](trial)
    args_copy['weight_decay'] = param_space['weight_decay'](trial)
    args_copy['learning_rate'] = param_space['learning_rate'](trial)
    args_copy['lr_decay'] = param_space['lr_decay'](trial)
    args_copy['decay_step'] = param_space['decay_step'](trial)

    avg_val_loss = main(args_copy, label, val_method='cross', loss=loss)
    return avg_val_loss

if __name__ == "__main__":

    args = load_config() 
    args.update(json.load(open("config/syn_data_corr_random_corr/basic_arg.json", 'r')))
    args = args_add(args)
    args['model_name'] = 'PTO'
    args['phase'] = 'train'

    # # For real data, we do not early stop
    # args['early_stop'] = False

    # For synthetic data, we use early stop
    args['early_stop'] = True
    args['save_step'] = args['epochs'] + 1  # no log saving during tuning
    args['delta'] = 0.00005
    args['patience'] = 5

    param_space = {
        'hidden_size_pto': lambda trial: trial.suggest_categorical('hidden_size_pto', [32, 64, 128]),
        'embedding_size': lambda trial: trial.suggest_categorical('embedding_size', [0, 1, 5, 10, 15, 20]),
        'num_layers': lambda trial: trial.suggest_categorical('num_layers', [1, 2, 3]),
        'epochs': lambda trial: trial.suggest_categorical('epochs', [20, 30, 40]),
        'batch_size': lambda trial: trial.suggest_categorical('batch_size', [64, 128, 256]),
        'weight_decay': lambda trial: trial.suggest_categorical('weight_decay', [0, 1e-6, 1e-5, 1e-4]),
        'learning_rate': lambda trial: trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 1e-2]),
        'lr_decay': lambda trial: trial.suggest_categorical('lr_decay', [0.4, 0.6, 0.8]),
        'decay_step': lambda trial: trial.suggest_categorical('decay_step', [1, 3, 5])
    }

    study_D = optuna.create_study(direction='minimize')
    study_D.optimize(lambda trial: objective(args, trial, 'D', loss='MSE'), n_trials=50)
    # save all the results
    results = study_D.trials_dataframe()
    if not os.path.exists(args['tuning_save_path']):
        os.makedirs(args['tuning_save_path'])
    results.to_csv(args['tuning_save_path'] + 'PTO_structure_tuning_D.csv', index=False)
    # save the best parameter as a json file
    best_params = study_D.best_params
    if not os.path.exists(args['config_path']):
        os.makedirs(args['config_path'])
    with open(args['config_path'] + 'PTO_structure_best_params_D.json', 'w') as f:
        json.dump(best_params, f)

    study_L = optuna.create_study(direction='minimize')
    study_L.optimize(lambda trial: objective(args, trial, 'L', loss='MSE'), n_trials=50)
    # save all the results
    results = study_L.trials_dataframe()
    if not os.path.exists(args['tuning_save_path']):
        os.makedirs(args['tuning_save_path'])
    results.to_csv(args['tuning_save_path'] + 'PTO_structure_tuning_L.csv', index=False)
    # save the best parameter as a json file
    best_params = study_L.best_params
    if not os.path.exists(args['config_path']):
        os.makedirs(args['config_path'])
    with open(args['config_path'] + 'PTO_structure_best_params_L.json', 'w') as f:
        json.dump(best_params, f)