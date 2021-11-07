import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

    def forward(self, features, query, labels=None, labels_query=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        #batch_size = ( features.shape[0] - self.K ) // 2
        batch_size = ( features.shape[0] - self.K ) // 65  # if we use 8*8 grid as query -----!!!!!! 64+1
        #batch_size = ( features.shape[0] - self.K ) // 128 # if we use 8*8 grid as query -----!!!!!! 64+64
        

        labels = labels.contiguous().view(-1, 1)
        #mask = torch.eq(labels[:batch_size], labels.T).float().to(device) #--------!!!!!! (1) for "repeat" or original
        mask = torch.eq(labels[:batch_size*64:64], labels.T).float().to(device) #------!!!!! (2) for "repeat_interleave"
        #mask = torch.eq(labels_query.contiguous().view(-1, 1), labels.T).float().to(device) #------!!!!! (3) take original query label

        # compute logits
        #anchor_dot_contrast = torch.div(
        #    torch.matmul(features[:batch_size], features.T),
        #    self.temperature)
        anchor_dot_contrast = torch.div(
            torch.matmul(query, features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        # logits_mask = torch.scatter( #---------------------------!!!!! (1) q and query same size
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size).view(-1, 1).to(device),

        #     0
        # )
        logits_mask = torch.scatter( #---------------------------!!!!! (2) q and query different (pixel q)
            torch.ones_like(mask),
            0,
            torch.arange(batch_size).repeat_interleave(64).view(1,-1).to(device),

            0
        )
        mask = mask * logits_mask

        # add ground truth 
        #one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32) #--------!!!!!! (1) for "repeat" or original
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size*64:64,].view(-1,), num_classes=self.num_classes).to(torch.float32) #------!!!!! (2) for "repeat_interleave"
        #one_hot_label = torch.nn.functional.one_hot(labels_query.contiguous().view(-1,), num_classes=self.num_classes).to(torch.float32) #------!!!!! (3) take original query label
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
