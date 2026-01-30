from config.config_loader import load_config, args_add
from PB_beta_tuning import load_pto_model
from train import load_basic_data
from env.simulation import simulator
import json
import warnings
warnings.filterwarnings("ignore")


def path_data_process(dataset, args, save=True):

    best_params_D = json.load(open(args['config_path']+"PTO_structure_best_params_D.json", "r"))
    best_params_L = json.load(open(args['config_path']+"PTO_structure_best_params_L.json", "r"))
    pto_model_D, pto_model_L = load_pto_model(args, dataset, best_params_D, best_params_L)
    best_beta = json.load(open(args['config_path'] + f"PB_best_params_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{args['train_seed']}.json", "r"))
    args.update(best_beta)

    pdp = simulator(args)
    instock_train, tildeD_train, PIL_train, Lnoise_train = pdp.simulate(dataset, pto_model=[pto_model_D, pto_model_L])
    path_data = pdp.generate_path_data(instock_train, tildeD_train, PIL_train, Lnoise_train, save=save)

    return path_data


if __name__ == "__main__":

    args = load_config()
    args.update(json.load(open("config/syn_data_corr_random_corr/basic_arg.json", "r")))
    args = args_add(args)
    dataset = load_basic_data(args)

    args['model_name'] = "PPB"
    args['phase'] = "train"
    path_data = path_data_process(dataset, args)