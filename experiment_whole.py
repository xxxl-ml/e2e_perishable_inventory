from config.config_loader import load_config, args_add
from basic_data_process import basic_data_process
from utils.data_loader import CustomDataset
from train import pto_train, ete_train, load_basic_data, load_path_data
from PB_beta_tuning import PB_tuning
from path_data_process import path_data_process
from ETE_lmbda_tuning import ETE_tuning
from ETE_boosting_tuning import ETE_boosting_tuning
from testing import pto_test, ete_test
import json
import pandas as pd
import numpy as np
import warnings
import os
import torch
warnings.filterwarnings("ignore")

'''
Note: this file includes the main process of the experiment follow:
1. generate the basic data (features, future sequqnces, etc.)
2. train the PTO model
3. tuning the PB model
4. generate the path data (like the constraint sampling, use a simple solution to get the sampled states) using tuned PPB
5. tuning the lmbda parameters for ETE models
6. train the ETE models
7. tuning the boosting gamma for ETE_PIL
8. test the models
9. save the results

There is other necessary process we do not include in this file:
- generate the orignal data, see the "data" folder.
- tune the PTO and ETE structure, see the "PTO_structure_tuning.py" and "ETE_structure_tuning.py". 
You may need to complete the above two steps and see if there is orignal data file in the "data" folder and tuned structure config file in the "config" folder.
Then, you can run this file to conduct the experiment.
'''


if __name__ == "__main__":

    seed_num = 1

    # load the configuration
    data_configration = "syn_data_corr_random_corr"   #---changing the data configuration here---"

    model_lst = ["PB", "PPB", "ETE_order_DLreg", "ETE_target_PILreg_noinstock", "ETE_target_PILreg_noinstock_boosting"]
    # model_lst = ["ETE_taget_PILreg_noinstock"]

    for perish in [4]:   #---changing the sensitivity parameter here---"
        expected_measure_list = np.zeros((seed_num, len(model_lst), 7))

        for seed in range(seed_num):

            args = load_config()
            args.update(json.load(open("config/" + data_configration + "/basic_arg.json", "r")))
            args = args_add(args)
            args['data_seed'] = seed  # only one dataset
            args['train_seed'] = seed
            args['perish'] = perish
            args['lmd_d'] = 0.0
            args['lmd_l'] = 0.0

            # generate the basic data
            if (seed > 0):
                basic_data = basic_data_process(args, save=True)
                dataset = CustomDataset(basic_data, args)
            else:
                dataset = load_basic_data(args)

            # train the PTO model
            train_time_lst = []
            if (seed > -1):
                args['phase'], args['model_name'], args['delta'], args['patience'] = "train", "PTO", 0.00005, 5
                model_D, model_L, train_time_D, train_time_L = pto_train(dataset, args)
                train_time_lst.append(["PTO_D", train_time_D])
                train_time_lst.append(["PTO_L", train_time_L])

            # tuning the PPB
            if (seed > -1):
                args['phase'] = "train"
                args['model_name'] = "PB_tuning"
                loss_list, best_beta = PB_tuning(args, beta_range=np.linspace(-0.5, 0.5, 21), dataset=dataset)

            # generate the path data
            if (seed > -1):
                args['model_name'] = "PPB"
                path_data = path_data_process(dataset, args, save=False)
                dataset.add_path_data(path_data)
            # else:
            #     dataset = load_path_data(args, dataset)

            # tuning the ETE models
            ete_lst = ["ETE_order_DLreg", "ETE_target_PILreg_noinstock"]
            args['phase'], args['delta'], args['patience'] = "train", 0.0001, 3
            if (seed > -1):
                for model_name in ete_lst:
                    args['model_name'] = model_name
                    print(f"Start for {args['model_name']}")
                    if "DLreg" in model_name:
                        reg_name = ["lmd_d", "lmd_l"]
                        reg_range = [[1.0, 2.5], [0.01, 0.1, 0.25]]
                    elif "PILreg" in model_name:
                        reg_name = ["lmd_pil0", "lmd_pil1"]
                        reg_range = [[1.0, 2.5], [0.1, 0.5, 1.0]]
                    loss_list, best_reg1, best_reg2, best_model, best_training_time = ETE_tuning(dataset, reg_name, reg_range, args, val_method="direct")
                    train_time_lst.append([args['model_name'], best_training_time])
                    print(f"model_name: {args['model_name']}, best_{reg_name[0]}: {best_reg1}, best_{reg_name[1]}: {best_reg2}")
                    torch.cuda.empty_cache()

            # train the ETE models
            if (seed > -1):
                ete_lst = ["ETE_order_DLreg", "ETE_target_PILreg_noinstock"]
                args['phase'], args['delta'], args['patience'] = "train", 0.0001, 3
                for model_name in ete_lst:
                    args['model_name'] = model_name
                    print("model_name: ", model_name)
                    ete_model, train_time = ete_train(dataset, args)     
                    train_time_lst.append([model_name, train_time])
                    torch.cuda.empty_cache()

            # tuning the ETE boosting gamma for ETE_target_PILreg_noinstock
            if (seed > -1):
                args['model_name'] = "ETE_target_PILreg_noinstock_boosting_tuning"
                args['phase'] = "train"
                gamma_range = np.linspace(0.8, 1.4, 13)
                loss_list, best_gamma = ETE_boosting_tuning(args, gamma_range, dataset=dataset)
                print(f"model_name: {args['model_name']}, best_gamma: {best_gamma}")

            # save the training time
            if (seed > -1):
                train_time_df = pd.DataFrame(train_time_lst, columns=["model_name", "training_time"])
                if not os.path.exists(args['test_save_path']):
                    os.makedirs(args['test_save_path'])
                train_time_df.to_csv(args['test_save_path'] + f"training_time_seed{seed}.csv", index=False)

            # test the models
            measures = []
            args['phase'] = "test"

            # load the test dataset
            if (seed > -1):

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
                results.insert(0, "model_name", model_lst)
                results.to_csv(args['test_save_path'] + f"point_measure_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{seed}.csv", index=False)

            else:
                measures = pd.read_csv(args['test_save_path'] + f"point_measure_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}_seed{seed}.csv")
                measures = measures.values[:, 1:]
            
            expected_measure_list[seed] = np.array(measures)

        # save the mean and std of expected_measures in each seed_train
        expected_measure_mean = np.mean(expected_measure_list, axis=0)
        expected_measure_std = np.std(expected_measure_list, axis=0)
        results_mean = pd.DataFrame(expected_measure_mean, columns=["TC", "HC", "BC", "OC", "SO", "OD", "evaluation_time"])
        results_std = pd.DataFrame(expected_measure_std, columns=["TC", "HC", "BC", "OC", "SO", "OD", "evaluation_time"])
        results_mean.insert(0, "model_name", model_lst)
        results_std.insert(0, "model_name", model_lst)
        results_mean.to_csv(args['test_save_path'] + f"expected_measure_mean_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}.csv", index=False)
        results_std.to_csv(args['test_save_path'] + f"expected_measure_std_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}.csv", index=False)

        # save all the records
        with open(args['test_save_path'] + f"measure_all_period{args['period']}_perish{args['perish']}_b{args['b']}_theta{args['theta']}.txt", "w") as f:
            f.write(str(expected_measure_list))