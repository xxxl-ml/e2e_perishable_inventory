'''
projected inventory level calculating layer (no learning)
only need to calculate the leadtime outdating and demand
output the projected inventory level (PIL) at arrival point (or from arrival point to perish point) for each sample
'''
import torch
import torch.nn as nn


class PILCAL(nn.Module):
    def __init__(self, MAX_LEAD, pil_seqlen, L_min, L_max):
        super(PILCAL, self).__init__()
        self.pil_seqlen = pil_seqlen
        self.total_len = MAX_LEAD + pil_seqlen - 1  # PIL not including the current demand
        self.L_min, self.L_max = float(L_min), float(L_max)

    def forward(self, instock, futured_pred, lead_pred):

        # calculating cumulative outdate quantity in each period within the leadtime
        outdate_lst = []
        futured_cumsum = torch.cumsum(futured_pred, dim=1)
        instock_cumsum = torch.cumsum(instock, dim=1)
        for t in range(self.total_len):
            if t == 0:
                outdate_lst.append(torch.relu(instock_cumsum[:, t:t+1] - futured_cumsum[:, t:t+1]))
            else:
                outdate_lst.append(torch.max(instock_cumsum[:, t:t+1] - futured_cumsum[:, t:t+1], outdate_lst[-1]))
        outdate_cumsum = torch.concat(outdate_lst, dim=1)

        bandwidth = 0.3
        if self.L_max != self.L_min:
            lead_pred_scale = (lead_pred * (self.L_max - self.L_min) + self.L_min).unsqueeze(2)
        else:
            lead_pred_scale = (lead_pred * self.L_max).unsqueeze(2)
            
        lead_pred_scale = torch.clamp(lead_pred_scale, min=self.L_min, max=self.L_max)
        # lead_pred_scale = torch.clamp(lead_pred_scale, min=0.8*self.L_min, max=1.2*self.L_max)
        # if torch.rand(1) < 0.1: print("lead_pred_scale: ", lead_pred_scale)
        # indices start from 1, when lead_pred=1, the consumed demand and outdating is only the first element
        indices = torch.arange(1, self.total_len+1, device=lead_pred.device).float().unsqueeze(0).expand(futured_pred.shape[0], -1)
        
        # use guassian kernel to make indexing differentiable
        indices_expanded = indices.unsqueeze(2)  # Shape: (batch_size, total_len, 1)
        k_values = torch.arange(self.pil_seqlen, device=futured_pred.device).view(1, 1, -1)  # Shape: (1, 1, pil_seqlen)

        # Calculate weights for all k (age after arrival) at once
        weights_un = torch.exp(-torch.pow((indices_expanded - (lead_pred_scale + k_values)) / bandwidth, 2) / 2) / bandwidth
        weights = weights_un / torch.sum(weights_un, dim=1, keepdim=True)  # Shape: (batch_size, total_len, pil_seqlen)

        # Calculate leadtime cumulative sums
        lead_d_cumsum = torch.sum(futured_cumsum.unsqueeze(2) * weights, dim=1)  # Shape: (batch_size, pil_seqlen)
        lead_o_cumsum = torch.sum(outdate_cumsum.unsqueeze(2) * weights, dim=1)  # Shape: (batch_size, pil_seqlen)

        # PIL = initial_inv - leadtime_d - leadtime_outdate
        pil = instock_cumsum[:, -1:] - lead_d_cumsum - lead_o_cumsum  # Shape: (batch_size, pil_seqlen)

        return pil