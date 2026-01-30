import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

# data flatten to the shape: (dc_sku_num * date_len(, seq_len), dim)
def data_flatten(data):
    data_flat = data.reshape(data.shape[0] * data.shape[1], *data.shape[2:])
    return data_flat

# data unflatten to the shape: (dc_sku_num, date_len(, seq_len), dim)
def data_unflatten(data, dc_sku_num, date_len):
    data_unflat = data.reshape(dc_sku_num, date_len, *data.shape[1:])
    return data_unflat

# data split into train dataset and test dataset, input data is unflattened
# the input is a data dictionary, the output is two data dictionaries for train and test
def data_split(data, test_len):
    train_data, test_data = {}, {}

    for key in data.keys():
        if data[key].shape[1] < test_len:
            raise ValueError(f"The data length for key '{key}' is smaller than test_len ({test_len}).")
        train_data[key] = data[key][:, :-test_len]
        test_data[key] = data[key][:, -test_len:]

    return train_data, test_data


class CustomDataset:

    def __init__(self, basic_data, args, existing_normalizer=None):

        # basic data including: sale, lead, cross, oth, dc, sku, L, D data, select the needed data
        self.device = args['device']
        
        # turn the data to torch.float32
        for key in basic_data.keys():
            basic_data[key] = basic_data[key].float()

        # split the basic data into train and test dataset
        self.feas, self.labels = ["sale", "lead", "cross", "oth", "dc", "sku"], ["L", "D"]
        self.train_set, self.test_set = data_split(basic_data, test_len=args['test_len'])

        # scale the basic feature data
        if args['fea_scale'] == "log":
            for name in self.feas:
                if name != "dc" and name != "sku":
                    self.train_set[name] = torch.log1p(self.train_set[name])
                    self.test_set[name] = torch.log1p(self.test_set[name])
        if args['fea_scale'] == "cbrt":
            for name in self.feas:
                if name != "dc" and name != "sku":
                    self.train_set[name] = torch.sign(self.train_set[name]) * torch.pow(torch.abs(self.train_set[name]), 1.0 / 3.0)
                    self.test_set[name] = torch.sign(self.test_set[name]) * torch.pow(torch.abs(self.test_set[name]), 1.0 / 3.0)

        self.normalizer = {}
        if existing_normalizer is not None:
            self.normalizer = existing_normalizer
            
        # normalize the basic data
        for name in self.feas:
            if name != "dc" and name != "sku":
                data_max = self.train_set[name].reshape(-1, self.train_set[name].shape[-1]).max(dim=0)[0]
                data_min = self.train_set[name].reshape(-1, self.train_set[name].shape[-1]).min(dim=0)[0]
                if existing_normalizer is None:
                    self.normalizer[name] = {"max": data_max, "min": data_min}
                if name == "lead" and data_max[0]==data_min[0]:  # constant leadtime
                    self.train_set[name] = self.train_set[name] / self.normalizer[name]["max"]
                    self.test_set[name] = self.test_set[name] / self.normalizer[name]["max"]
                else:
                    data_max = torch.where(data_max==data_min, data_max+1, data_max)
                    if existing_normalizer is None:
                        self.normalizer[name] = {"max": data_max, "min": data_min}
                    self.train_set[name] = (self.train_set[name] - self.normalizer[name]["min"]) / (self.normalizer[name]["max"] - self.normalizer[name]["min"])
                    self.test_set[name] = (self.test_set[name] - self.normalizer[name]["min"]) / (self.normalizer[name]["max"] - self.normalizer[name]["min"])
            else:
                self.train_set[name] = self.train_set[name].long()
                self.test_set[name] = self.test_set[name].long()
                
        if args['normalize'] == "global":
            combined_L = torch.cat([self.train_set["L"].reshape(-1), self.test_set["L"].reshape(-1)])
            self.L_max, self.L_min = float(torch.max(combined_L)), float(torch.min(combined_L))
            combined_D = torch.cat([self.train_set["D"].reshape(-1), self.test_set["D"].reshape(-1)])
            self.D_max, self.D_min = float(torch.max(combined_D)), float(torch.min(combined_D))
        else:
            self.L_max, self.L_min = float(torch.max(self.train_set["L"])), float(torch.min(self.train_set["L"]))
            self.D_max, self.D_min = float(torch.max(self.train_set["D"])), float(torch.min(self.train_set["D"]))

        if existing_normalizer is None:
            self.normalizer["L"] = {"max": self.L_max, "min": self.L_min}
            self.normalizer["D"] = {"max": self.D_max, "min": self.D_min}

        self.train_set["L"] = self.train_set["L"] / self.normalizer["L"]["max"] if self.normalizer["L"]["max"]==self.normalizer["L"]["min"] else (self.train_set["L"] - self.normalizer["L"]["min"]) / (self.normalizer["L"]["max"] - self.normalizer["L"]["min"])
        self.test_set["L"] = self.test_set["L"] / self.normalizer["L"]["max"] if self.normalizer["L"]["max"]==self.normalizer["L"]["min"] else (self.test_set["L"] - self.normalizer["L"]["min"]) / (self.normalizer["L"]["max"] - self.normalizer["L"]["min"])
        self.train_set["D"] = (self.train_set["D"] - self.normalizer["D"]["min"]) / (self.normalizer["D"]["max"] - self.normalizer["D"]["min"])
        self.test_set["D"] = (self.test_set["D"] - self.normalizer["D"]["min"]) / (self.normalizer["D"]["max"] - self.normalizer["D"]["min"])

        # to device: the test data always on cpu
        for key in self.train_set.keys():
            self.train_set[key] = self.train_set[key].to(args['device'])

        self.have_path = False


    def get_normalizer(self):

        return self.normalizer


    def add_path_data(self, path_train):

        # including instock, PIL, Dh, Db, h_cost, b_cost and theta_cost data
        path_name = ["instock", "PIL", "Dh", "Db", "h", "b", "theta"]  # the h/b/theta coefficient data no need to normalize

        # all the labels and loss function related labels are scaled by a same scaler to keep a well defined loss function
        for name in path_name[:4]:
            self.train_set[name] = (path_train[name] - self.normalizer["D"]["min"]) / (self.normalizer["D"]["max"] - self.normalizer["D"]["min"])
        for name in path_name[4:]:
            self.train_set[name] = path_train[name]

        # to device
        for key in path_name:
            self.train_set[key] = self.train_set[key].to(self.device)
        
        self.have_path = True


    def get_data(self, keys, dc_sku_idx=None, phase="train", device=None, flatten=True, train_ratio=1.0):

        if device is None:
            device = self.device
        train_len = int(self.train_set["sale"].shape[1] * train_ratio)  # the length of the training data

        if dc_sku_idx is not None:
            data = []
            if phase == "train":
                for key in keys:
                    data.append(self.train_set[key][dc_sku_idx][:train_len].to(device))
            elif phase == "test":
                for key in keys:
                    data.append(self.test_set[key][dc_sku_idx].to(device))
            else:
                raise ValueError(f"Phase '{phase}' is not supported.")
            return data

        else:
            if flatten:
                flat_data = []
                if phase == "train":
                    for key in keys:
                        flat_data.append(data_flatten(self.train_set[key][:,:train_len]).to(device))
                elif phase == "test":
                    for key in keys:
                        flat_data.append(data_flatten(self.test_set[key]).to(device))
                else:
                    raise ValueError(f"Phase '{phase}' is not supported.")
                return flat_data
            else:
                unflat_data = []
                if phase == "train":
                    for key in keys:
                        unflat_data.append(self.train_set[key][:,:train_len].to(device))
                elif phase == "test":
                    for key in keys:
                        unflat_data.append(self.test_set[key].to(device))
                else:
                    raise ValueError(f"Phase '{phase}' is not supported.")
                return unflat_data
            

    def get_dataloader(self, keys, batch_size, have_val=True, train_ratio=1.0):  # get the dataloader for the train phase, dc_sku_idx is always None

        if have_val:  # let the last 20% time series data as validation set
            loader_set = self.get_data(keys, flatten=False, train_ratio=train_ratio)
            train_data, val_data = [], []
            for item in loader_set:  # leave the median 10% to prevent the data leakage
                train_data.append(data_flatten(item[:, :-int(item.shape[1]*0.3)]))
                val_data.append(data_flatten(item[:, -int(item.shape[1]*0.2):]))
            del loader_set
            torch.cuda.empty_cache()
            train_data = DataLoader(TensorDataset(*train_data), batch_size=batch_size, shuffle=True)
            val_data = DataLoader(TensorDataset(*val_data), batch_size=batch_size, shuffle=False)
            return train_data, val_data
        else:
            loader_set = TensorDataset(*self.get_data(keys, train_ratio=train_ratio))
            train_data = DataLoader(loader_set, batch_size=batch_size, shuffle=True)
            return train_data
        

    def get_k_fold(self, k, keys, device=None):  # phase is always "train", flatten is always True

        if device is None:
            device = self.device

        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        flat_data = {}
        for key in keys:
            flat_data[key] = data_flatten(self.train_set[key]).to(device)
        
        for train_index, val_index in kfold.split(flat_data[keys[0]]):
            train_data, val_data = [], []
            for key in keys:
                train_data.append(flat_data[key][train_index])
                val_data.append(flat_data[key][val_index])
            yield train_data, val_data