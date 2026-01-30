'''
Conduct data processing for the pivoted data
generate the basic data
and use simulation to generate the path data for ETE input
save the data
'''

from config.config_loader import load_config, args_add
from utils.basic_data import basic_data_parser
import json

def basic_data_process(args, save=True):
    
    bpd = basic_data_parser(args)
    basic_data = bpd.generate_data(save=save)

    return basic_data

if __name__ == "__main__":

    args = load_config()
    args.update(json.load(open("config/syn_data_corr_random_corr/basic_arg.json", "r")))
    for seed in range(1):
        args['seed'] = seed
        args = args_add(args)

        basic_data = basic_data_process(args)