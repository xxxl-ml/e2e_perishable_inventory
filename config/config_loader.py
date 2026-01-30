import argparse
import json
import os
import pandas as pd
import numpy as np

def load_config():

    parser = argparse.ArgumentParser(description="All Things Configuration")
    
    # problem setting
    parser.add_argument("--period", type=int, default=4, help="period")
    parser.add_argument("--perish", type=int, default=7, help="life time of the product")
    parser.add_argument("--b", type=float, default=10.0, help="backorder cost")
    parser.add_argument("--h", type=float, default=1.0, help="holding cost")
    parser.add_argument("--theta", type=float, default=10.0, help="outdating cost")

    parser.add_argument("--test_len", type=int, default=30, help="test data length")
    parser.add_argument("--l_histlen", type=int, default=7, help="history length used for leadtime prediction")
    parser.add_argument("--d_histlen", type=int, default=30, help="history length used for demand prediction")
    parser.add_argument("--cross_histlen", type=int, default=7, help="cross term history length used for demand and leadtime prediction")
    parser.add_argument("--DL_corr", type=bool, default=False, help="belief of correlation between demand and leadtime")
    parser.add_argument("--MAX_LEAD", type=int, default=9, help="maximum leadtime")
    parser.add_argument("--MAX_PERISH", type=int, default=7, help="maximum perish time")

    # data path
    parser.add_argument("--data_path", type=str, default="data/real_data/", help="data path: sales, lead, instock and features")
    parser.add_argument("--data_save_path", type=str, default="data/real_data/data_clean/", help="save path for basic and path data after data processing")
    parser.add_argument("--model_save_path", type=str, default="model_saved/real_data/", help="save path for trained model")
    parser.add_argument("--config_path", type=str, default="config/real_data/", help="save path for best parameters")
    parser.add_argument("--tuning_save_path", type=str, default="tuning_result/real_data/", help="save path for tuning results")
    parser.add_argument("--test_save_path", type=str, default="test_result/real_data/", help="save path for test results")
    parser.add_argument("--extra_sale_cvt", type=bool, default=False, help="whether to use extra sale covariate")
    parser.add_argument("--extra_lead_cvt", type=bool, default=False, help="whether to use extra lead covariate")

    # data processing
    parser.add_argument("--fea_scale", type=str, default="cbrt", choices=['log', 'cbrt', None], help="feature scaling method")
    parser.add_argument("--normalize", type=str, default="global", choices=['global', 'training', None], help="global")
    parser.add_argument("--noise_sample_num", type=int, default=100, help="sample number for noise for ETE training")
    parser.add_argument("--dl_sample_num", type=int, default=100, help="sample number for demand and leadtime for PB policy")
    parser.add_argument("--beta_adjust", type=float, default=0, help="balancing coefficient adjustment for parameterized PB")  # tuning
    parser.add_argument("--boosting_gamma", type=float, default=1.0, help="boosting gamma for ETE decision")  # tuning

    # model setting: model structure arguments
    parser.add_argument("--model_name", type=str, default="PTO", choices=["PTO"], help="model name")
    parser.add_argument("--hidden_size_pto", type=int, default=64, help="the PTO model: hidden size of LSTM prediction")  # tuning
    parser.add_argument("--hidden_size_ete", type=list, default=[64, 32, 128, 128], help="the ETE model: hidden size of demand/lead predction layer, other feature fc layer and concat layer")  # tuning
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers of LSTM")  # tuning
    parser.add_argument("--embedding_size", type=int, default=10, help="embedding size of dc and sku id")  # tuning

    # model setting: training arguments
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"], help="device")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")  # tuning
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")  # tuning
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")  # tuning
    parser.add_argument("--decay_step", type=int, default=5, help="decay step")  # tuning
    parser.add_argument("--lr_decay", type=float, default=0.61, help="decay rate")  # tuning

    parser.add_argument("--epochs", type=int, default=60, help="number of epochs")  # tuning if no early stopping
    parser.add_argument("--early_stop", type=bool, default=False, help="whether to use early stopping")
    parser.add_argument("--log_step", type=int, default=1, help="log step")
    parser.add_argument("--save_step", type=int, default=5, help="save step")
    parser.add_argument("--patience", type=int, default=5, help="patience for early stopping")
    parser.add_argument("--delta", type=float, default=0.0001, help="delta for early stopping")

    # model setting: regularization arguments
    parser.add_argument("--lmd_noise", type=float, default=1.0, help="lambda for noise loss")  # tuning
    parser.add_argument("--lmd_d", type=float, default=1.0, help="lambda for demand prediction regularization")  # tuning
    parser.add_argument("--lmd_l", type=float, default=0.01, help="lambda for leadtime prediction regularization")  # tuning
    parser.add_argument("--lmd_pil0", type=float, default=1.0, help="lambda for arrival point PIL prediction regularization")  # tuning
    parser.add_argument("--lmd_pil1", type=float, default=0.5, help="lambda for PIL sequence prediction regularization")  # tuning
    
    # other general arguments
    parser.add_argument("--phase", type=str, default="train", help="train or test")
    parser.add_argument("--data_seed", type=int, default=0, help="dataset seed")
    parser.add_argument("--train_seed", type=int, default=0, help="training seed")
    parser.add_argument("--config", type=str, default=None, help="Config File Path")

    # if there is a config file, first load it and update with the command line arguments
    args = parser.parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            params = json.load(f)
            params.update(vars(args))
    else:
        params = vars(args)

    return params

def args_add(args):

    lead_pivoted = pd.read_csv(args['data_path'] + f"lead_seed{args['data_seed']}.csv", index_col=0)
    dc_sku_list = lead_pivoted.columns
    dc_list = np.unique([int(dc_sku[2:5]) for dc_sku in dc_sku_list])
    sku_list = np.unique([int(dc_sku[-3:]) for dc_sku in dc_sku_list])
    dc_num, sku_num = max(dc_list), max(sku_list)
    args['MAX_LEAD'] = int(np.max(lead_pivoted.max()))
    args['num_dc'], args['num_sku'] = dc_num, sku_num

    return args