import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DL_pred import DL_LSTM
from model.PIL_cal import PILCAL


class ETE(nn.Module):
    def __init__(self, args, sizes, L_min, L_max):
        super(ETE, self).__init__()

        self.model_name = args['model_name']
        self.MAX_LEAD = args['MAX_LEAD']
        self.is_noinstock = "noinstock" in args['model_name']
        self.is_target = "target" in args['model_name']
        self.is_PILreg = "PILreg" in args['model_name']

        # demand forcast
        self.demand_lstm = DL_LSTM(sizes["salefea_size"], sizes["hidden_size"][0], args['MAX_LEAD']+args['perish'], args['embedding_size'], args['num_dc'], args['num_sku'], args['num_layers'])

        # leadtime forcast
        self.lead_lstm = DL_LSTM(sizes["leadfea_size"], sizes["hidden_size"][1], 2, args['embedding_size'], args['num_dc'], args['num_sku'], args['num_layers'])

        # cross term, oth input layer
        self.others_fc = nn.Linear(sizes["cross_size"]+sizes["oth_size"], sizes["hidden_size"][2])
        self.dropout = nn.Dropout(0.5)
        if not self.is_noinstock:
            self.instock_fc = nn.Linear(sizes["instock_size"], sizes["hidden_size"][2])

        # pil calculating layer
        if self.is_target:  # when the model output target level, the pil_cal layer is needed
            if self.is_PILreg:
                self.pil_cal = PILCAL(args['MAX_LEAD'], args['perish'], L_min, L_max)
            else:
                self.pil_cal = PILCAL(args['MAX_LEAD'], 1, L_min, L_max)

        # concate linear layer
        if self.is_noinstock:
            self.concat_fc1 = nn.Linear(sizes["hidden_size"][0] + sizes["hidden_size"][1] + sizes["hidden_size"][2], sizes["hidden_size"][3])
        else:
            self.concat_fc1 = nn.Linear(sizes["hidden_size"][0] + sizes["hidden_size"][1] + 2 * sizes["hidden_size"][2], sizes["hidden_size"][3])
        self.concat_fc2 = nn.Linear(sizes["hidden_size"][3], sizes["output_size"])

        self._initialize_weights()  # initialize all the parameters

    def _initialize_weights(self):

        for param in self.others_fc.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)
        if not self.is_noinstock:
            for param in self.instock_fc.parameters():
                nn.init.normal_(param, mean=0.0, std=0.01)
        for param in self.concat_fc1.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)
        for param in self.concat_fc2.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)

    def forward(self, sale, lead, cross, oth, instock, dc, sku):

        # demand forcast
        sale_pred, sale_hidden = self.demand_lstm(sale, dc, sku)

        # leadtime forcast
        lead_pred, lead_hidden = self.lead_lstm(lead, dc, sku)

        # cross term and oth input
        crossoth = torch.cat((cross, oth), dim=1)
        crossoth_hidden = self.others_fc(crossoth)
        crossoth_hidden = self.dropout(crossoth_hidden)
        crossoth_hidden = F.relu(crossoth_hidden)

        # instock input
        if not self.is_noinstock:
            instock_hidden = self.instock_fc(instock)
            instock_hidden = F.relu(instock_hidden)
            crossoth_hidden = torch.cat((crossoth_hidden, instock_hidden), dim=1)

        # pil calculating
        if self.is_target:
            if self.is_PILreg:
                pil_out = self.pil_cal(instock, sale_pred[:, :-1], lead_pred[:,:1])
            else:
                pil_out = self.pil_cal(instock, sale_pred[:,:self.MAX_LEAD], lead_pred[:,:1])
        else:
            pil_out = None

        # concate linear layer
        x = torch.cat((sale_hidden, lead_hidden, crossoth_hidden), dim=1)
        x = self.concat_fc1(x)
        x = F.relu(x)
        out = self.concat_fc2(x)

        return out, pil_out, sale_pred, lead_pred