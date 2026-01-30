import torch
import torch.nn as nn

class MNVLoss(nn.Module):
    def __init__(self, model_name, lmd_d=1.0, lmd_l=0.01, lmd_pil0=1.0, lmd_pil1=0.5, val=False):
        super(MNVLoss, self).__init__()
        self.model_name = model_name
        self.val = val
        if self.val:
            self.lmd_d = 0
            self.lmd_l = 0
            self.lmd_pil0 = 0
            self.lmd_pil1 = 0
            self.lmd_penalty = 0
        else:
            self.lmd_d = lmd_d
            self.lmd_l = lmd_l
            self.lmd_pil0 = lmd_pil0
            self.lmd_pil1 = lmd_pil1
            self.lmd_penalty = 10 * max(lmd_d, lmd_l, lmd_pil0, lmd_pil1)
        self.mse_loss = nn.MSELoss()

        self.is_order = "order" in model_name
        self.is_noise = "noise" in model_name
        self.is_DLreg = "DLreg" in model_name
        self.is_PILreg = "PILreg" in model_name

    def forward(self, target_o, pil_o, demand_o, lead_o, demand_true, lead_true, pil_true, Dh, Db, h_cost, b_cost, theta_cost):
        # notice that output and D are all normalized here, but for gradient, the normalized coefficient does not matter

        # order_q shape: (batch_size, 1, 1)
        if self.is_order:
            order_q = target_o.unsqueeze(1)
        else:
            order_q = torch.relu(target_o - pil_o[:, :1]).unsqueeze(1)

        # Dh, Db, h, b shape: (batch_size, noise_sample_num, perish_len); theta shape: (batch_size, noise_sample_num, 1)
        if self.is_noise and (self.val == False):
            order_loss = torch.mean(torch.sum(h_cost[:, 1:, :] * torch.relu(order_q - Dh[:, 1:, :]) + b_cost[:, 1:, :] * torch.relu(Db[:, 1:, :] - order_q), dim=2) \
                + torch.sum(theta_cost[:, 1:, :] * torch.relu(order_q - Dh[:, 1:, -1:]), dim=2))
        else:
            order_loss = torch.mean(torch.sum(h_cost[:, :1, :] * torch.relu(order_q - Dh[:, :1, :]) + b_cost[:, :1, :] * torch.relu(Db[:, :1, :] - order_q), dim=2) \
                + torch.sum(theta_cost[:, :1, :] * torch.relu(order_q - Dh[:, :1, -1:]), dim=2))
        loss = order_loss.clone()
        
        if self.is_DLreg:
            d_loss = self.mse_loss(demand_o, demand_true[:, :demand_o.shape[1]])
            l_loss = self.mse_loss(lead_o, lead_true)
            loss += self.lmd_d * d_loss + self.lmd_l * l_loss
        else:
            d_loss = torch.zeros_like(order_loss)
            l_loss = torch.zeros_like(order_loss)

        if self.is_PILreg:
            d_loss = self.mse_loss(demand_o, demand_true[:, :demand_o.shape[1]])  # ensuring the pred demand in an appropriate range
            l_loss = self.mse_loss(lead_o, lead_true)  # ensuring the pred leadtime in an appropriate range
            pil_loss0 = self.mse_loss(pil_o[:, :1], pil_true[:,0,:1])
            pil_loss1 = self.mse_loss(pil_o[:, 1:], pil_true[:,0,1:])
            loss += self.lmd_d * d_loss + self.lmd_l * l_loss + self.lmd_pil0 * pil_loss0 + self.lmd_pil1 * pil_loss1
        else:
            pil_loss0 = torch.zeros_like(order_loss)
            pil_loss1 = torch.zeros_like(order_loss)

        if self.is_order == False:  # push back for the irrational leadtime prediction in PILCAL
            penalty_loss = self.lmd_penalty * torch.mean(torch.relu(lead_o[:,:1] - 1) + torch.relu(-lead_o[:,:1]))
            loss += penalty_loss

        return loss, order_loss, d_loss, l_loss, pil_loss0, pil_loss1
    

class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self, output, true):

        # true demand shape: (batch_size, 1)
        # output shape: (batch_size, 1)
        loss = self.quantile * torch.relu(true - output) + (1 - self.quantile) * torch.relu(output - true)
        return loss.mean()