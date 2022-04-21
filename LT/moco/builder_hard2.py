"""https://github.com/facebookresearch/moco"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import pdb

class NormedLinear_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048):
        super(NormedLinear_Classifier, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, *args):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


def flatten(t):
    return t.reshape(t.shape[0], -1)

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.2, mlp=False, feat_dim=2048, normalize=False, num_classes=1000):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, CAM=True, return_features=True)#----------------!!!!!!
        self.encoder_k = base_encoder(num_classes=dim, CAM=True, return_features=True)#---------------!!!!!!
        self.linear = nn.Linear(feat_dim, num_classes)
        self.linear_k = nn.Linear(feat_dim, num_classes)


        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            #self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_q.fc = nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp,kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(dim_mlp), nn.ReLU(), self.encoder_q.fc)#----!!!!
            #self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_k.fc)
            self.encoder_k.fc = nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp,kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(dim_mlp), nn.ReLU(), self.encoder_k.fc)#----!!!!

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # create the queue
        self.register_buffer("queue_global", torch.randn(K, dim))
        self.queue_global = nn.functional.normalize(self.queue_global, dim=0)
        
        self.register_buffer("queue_local", torch.randn(K, dim))
        self.queue_local = nn.functional.normalize(self.queue_local, dim=0)

        self.register_buffer("queue_p", torch.randn(K*4, 100)) #-----!!!!
        self.queue_p = nn.functional.normalize(self.queue_p, dim=1) #-----!!!!

        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # cross_entropy
        self.layer = -2 
        self.feat_after_avg_k = None
        self.feat_after_avg_q = None
        self._register_hook()

        self.normalize = normalize

    
    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]

            return children[self.layer]
        return None

    def _hook_k(self, _, __, output):
        self.feat_after_avg_k = flatten(output)
        if self.normalize: 
           self.feat_after_avg_k = nn.functional.normalize(self.feat_after_avg_k, dim=1)


    def _hook_q(self, _, __, output):
        self.feat_after_avg_q = flatten(output)
        if self.normalize:
           self.feat_after_avg_q = nn.functional.normalize(self.feat_after_avg_q, dim=1)


    def _register_hook(self):
        layer_k = self._find_layer(self.encoder_k)
        assert layer_k is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_k.register_forward_hook(self._hook_k)

        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_q.register_forward_hook(self._hook_q)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    #def _dequeue_and_enqueue(self, keys, labels):
    def _dequeue_and_enqueue(self, keys_local, keys_global, labels, atts):#----!!
        # gather keys before updating queue
        keys_local = concat_all_gather(keys_local)
        keys_global = concat_all_gather(keys_global)
        labels = concat_all_gather(labels)


        batch_size = keys_global.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity


        # replace the keys at ptr (dequeue and enqueue)
        self.queue_global[ptr:ptr + batch_size,:] = keys_global
        self.queue_local[ptr:ptr + batch_size,:] = keys_local
        self.queue_l[ptr:ptr + batch_size] = labels


        self.queue_p[ptr:ptr + batch_size,:] = atts #------!!!!!

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]


        return x_gather[idx_this], y_gather[idx_this]

    def _train(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, q_encoding = self.encoder_q(im_q)  # queries: NXWHXC , NxC #----!!!  w/ encoding output
        q_local = nn.functional.normalize(q, dim=1)

        q_global = q.mean(2) # [128,32,8*8] -> [128,32]
        q_global = nn.functional.normalize(q_global, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, labels, idx_unshuffle = self._batch_shuffle_ddp(im_k, labels)

            k, k_encoding = self.encoder_k(im_k)  # keys: NxC # queries: NxC #----!!!  w/ encoding output
            k_local = nn.functional.normalize(k, dim=1)

            k_global = k.mean(2) # [128,32,8*8] -> [128,32]
            k_global = nn.functional.normalize(k_global, dim=1)
            

            #q = q.mean(2) # [128,32,8*8] -> [128,32] # with grad??

            # selecting q
            # with torch.no_grad():
            #     score_map = self.linear(q_encoding.permute(0,2,3,1).reshape(128*8*8,64)).view(-1,8*8,self.linear.weight.shape[0]) # [128,64,8,8] -> [128,8*8,64] -> [128,8*8,100]                       .permute(0,3,1,2).reshape(128,100,8*8)  #----!!!!! CAM for q
                
            #     q_labels = score_map.argmax(2) #----!!!!! [128,8*8,100] -> [128,8*8]        .permute(1,2,0).reshape(-1)  ) [128,8,8] -> [8,8,128] -> [8*8*128]
            #     pix_ind = (q_labels!=labels.unsqueeze(1)).float() # true or false of target cls
            #     # center_mask = (pix_ind / pix_ind.sum(1).unsqueeze(1)).unsqueeze(-1) # normalize
            #     # q = torch.bmm(q,center_mask).squeeze()
                
            #     if 0 in pix_ind.sum(1):
            #         pix_ind = torch.randint(64,(128,1,1)) # (not RE)
            #         #max  (RE)
            #         #q_labels = torch.gather(score_map,2,labels.unsqueeze(-1).repeat(1,64).unsqueeze(-1)).squeeze() #[128,64,100] [128, 64]
            #         #_, pix_ind = q_labels.min(1) #max or min ind for target label
            #         #pix_ind = pix_ind.unsqueeze(1).unsqueeze(2) #[128] -> [128,1,1]
            #     else:
            #         # (1) random choose among true
            #         #tmp = [pp.nonzero()[torch.randint(pp.sum().long().item(),(1,))].item()   for pp in pix_ind] # random select
            #         #pix_ind = torch.LongTensor(tmp).unsqueeze(-1).unsqueeze(-1) # [128,1,1]
                    
            #         # (2) choose max / min CAM score (YAME implementation!)
            #         #q_labels = torch.gather(score_map,2,labels.unsqueeze(-1).repeat(1,64).unsqueeze(-1)).squeeze() #[128,64,100] [128, 64] #(2-1) max of GT cls
            #         q_labels,_ = score_map.max(2) # (2-2) max among all classes
            #         _, pix_ind = (q_labels*pix_ind).max(1) #max or min ind for target label
            #         pix_ind = pix_ind.unsqueeze(1).unsqueeze(2) #[128] -> [128,1,1]
                
            #     # (0) All Max not True
            #     # q_labels,_ = score_map.max(2) # (2-2) max among all classes
            #     # _, pix_ind = (q_labels).max(1) #max or min ind for target label
            #     # pix_ind = pix_ind.unsqueeze(1).unsqueeze(2) #[128] -> [128,1,1]
                

            ## q_labels = torch.gather(score_map,2,labels.unsqueeze(-1).repeat(1,64).unsqueeze(-1)).squeeze() #[128,64,100] [128, 64]
            ## _, pix_ind = q_labels.min(1) #max or min ind for target label
            ## pix_ind = pix_ind.unsqueeze(1).unsqueeze(2) #[128] -> [128,1,1]

            with torch.no_grad():
                score_map_k = self.linear(k_encoding.permute(0,2,3,1).reshape(128*8*8,64)).view(-1,8*8,self.linear.weight.shape[0]) # [128,64,8,8] -> [128,8*8,64] -> [128,8*8,100] CAM for k
                
                # max k
                k_labels = torch.gather(score_map_k,2,labels.unsqueeze(-1).repeat(1,64).unsqueeze(-1)).squeeze() #[128,64,100] [128, 64]
                _, pix_ind_k = k_labels.max(1) #max or min ind for target label
                pix_ind_k = pix_ind_k.unsqueeze(1).unsqueeze(2) #[128] -> [128,1,1]

                # True k
                # k_labels = score_map_k.argmax(2) #----!!!!! [128,8*8,100] -> [128,8*8]        .permute(1,2,0).reshape(-1)  ) [128,8,8] -> [8,8,128] -> [8*8*128]
                # pix_ind_k = (k_labels==labels.unsqueeze(1)).float() # true or false of target cls
                # if 0 in pix_ind_k.sum(1):
                #     pix_ind_k = torch.randint(64,(128,1,1)) # (not RE)
                #     # max k (RE)
                #     # k_labels = torch.gather(score_map_k,2,labels.unsqueeze(-1).repeat(1,64).unsqueeze(-1)).squeeze() #[128,64,100] [128, 64]
                #     # _, pix_ind_k = k_labels.max(1) #max or min ind for target label
                #     # pix_ind_k = pix_ind_k.unsqueeze(1).unsqueeze(2) #[128] -> [128,1,1]
                # else:
                #      # (1) random choose among true (RandomTruek)
                #     tmp = [pp.nonzero()[torch.randint(pp.sum().long().item(),(1,))].item()   for pp in pix_ind_k] # random select
                #     pix_ind_k = torch.LongTensor(tmp).unsqueeze(-1).unsqueeze(-1) # [128,1,1]
                  
                #     # (2) choose max / min CAM score (MaxTruek) (YAME implementation!)
                #     # #k_labels = torch.gather(score_map,2,labels.unsqueeze(-1).repeat(1,64).unsqueeze(-1)).squeeze() #[128,64,100] [128, 64] #(2-1) max of GT cls (MaxTruek)
                #     # k_labels,_ = score_map_k.max(2) # (2-2) max among all classes (AlMaxTrueK)
                #     # _, pix_ind_k = (k_labels*pix_ind_k).max(1) #max or min ind for target label
                #     # pix_ind_k = pix_ind_k.unsqueeze(1).unsqueeze(2) #[128] -> [128,1,1]

                # # (0) All Max not True
                # # k_labels,_ = score_map_k.max(2) # (2-2) max among all classes
                # # _, pix_ind_k = (k_labels).max(1) #max or min ind for target label
                # # pix_ind_k = pix_ind_k.unsqueeze(1).unsqueeze(2) #[128] -> [128,1,1]


            #q = torch.gather(q,2,pix_ind.repeat(1,32,1).cuda()).squeeze() # choose q based on the ind (B number of indexes)
            
            
            # attentional choose k based on q
            # att = torch.matmul(q,k.permute(1,0,2).reshape(32,-1)).reshape(128,128,64) #[128,32]X[32,128*64] -> [128,128*64]->[128,128,8*8]
            # #for ii in range(128):
            # #    att[ii,ii,:]=-100
            # Temp =2
            # att,_ = att.max(0) #att = att.mean(0) # better than mean?
            # att = (att/Temp).softmax(-1) # temperature??            
            # k = torch.bmm(k,att.unsqueeze(2)).squeeze() #[128,32,8*8] X [128,8*8,1] -> [128,32,1] ->  [128,32]

            k_local = torch.gather(k_local,2,pix_ind_k.repeat(1,32,1).cuda()).squeeze() # choose k based on the ind (B number of indexes)

            # # attentional choose q based on k
            #att = torch.matmul(k,q.permute(1,0,2).reshape(32,-1)).reshape(128,128,64) #[128,32]X[32,128*64] -> [128,128*64]->[128,128,8*8]
            att = torch.matmul(self.queue_local.clone().detach(),q.permute(1,0,2).reshape(32,-1)).reshape(self.K,128,64) #[128,32]X[32,128*64] -> [128,128*64]->[128,128,8*8]
            #for ii in range(128):
            #   att[ii,ii,:]=-100
            Temp =1
            att,_ = att.max(0) #att = att.mean(0) # better than mean?
            att = (att/Temp).softmax(-1) # temperature??            
            ##q = torch.bmm(q,att.unsqueeze(2)).squeeze() #[128,32,8*8] X [128,8*8,1] -> [128,32,1] ->  [128,32] not w/Grad
            
            
            #att_label = (torch.gather(score_map_k.softmax(1),1,pix_ind_k.repeat(1,1,100).cuda()).squeeze()>0.3).float() # [B*100] (attribute for BCE loss)
            att_k = nn.functional.normalize(torch.gather(score_map_k.softmax(1),1,pix_ind_k.repeat(1,1,100).cuda()).squeeze(),dim=1) # [B*100] # for self attribute (with selection. general)
            #att_label = torch.gather(score_map_k,1,pix_ind_k.repeat(1,1,100).cuda()).squeeze().softmax(1) # [B*100] # for local k ! to train global q (attribute global)
            #att_label = score_map_k.softmax(1) #for all  k to all q (attribute all)
            #att_label = (torch.mm(k,self.cluster_centers.t().cuda())/0.05) #.argmax(1) #cluster_ids_x[:128].cuda() # for clustering!

            #k = torch.mm(k.reshape(-1,64),center_mask).reshape(-1,32)
            #q = torch.mm(q.reshape(-1,64),center_mask).reshape(-1,32)

            # undo shuffle
            k_local, labels = self._batch_unshuffle_ddp(k_local, labels, idx_unshuffle)
        
        q_local = torch.bmm(q_local,att.unsqueeze(2)).squeeze() #[128,32,8*8] X [128,8*8,1] -> [128,32,1] ->  [128,32] w/Grad

        # compute logits 
        logits_q = self.linear(self.feat_after_avg_q)


        score_map = self.linear(q_encoding.permute(0,2,3,1).reshape(128*8*8,64)).view(-1,8*8,self.linear.weight.shape[0]) # [128,64,8,8] -> [128,8*8,64] -> [128,8*8,100]                       .permute(0,3,1,2).reshape(128,100,8*8)  #----!!!!! CAM for q
        
        if False: # (1) random selection from CAM of max likelihood (self attention)
            q_labels = score_map.argmax(2) #----!!!!! [128,8*8,100] -> [128,8*8]        .permute(1,2,0).reshape(-1)  ) [128,8,8] -> [8,8,128] -> [8*8*128]
            pix_ind = (q_labels==logits_q.argmax(1).unsqueeze(1)).float() # true or false of target cls
            if 0 in pix_ind.sum(1):
                pix_ind = torch.randint(64,(128,1,1)) # (not RE)
                #max  (RE)
                #q_labels = torch.gather(score_map,2,labels.unsqueeze(-1).repeat(1,64).unsqueeze(-1)).squeeze() #[128,64,100] [128, 64]
                #_, pix_ind = q_labels.min(1) #max or min ind for target label
                #pix_ind = pix_ind.unsqueeze(1).unsqueeze(2) #[128] -> [128,1,1]
            else:
                tmp = [pp.nonzero()[torch.randint(pp.sum().long().item(),(1,))].item()   for pp in pix_ind] # random select
                pix_ind = torch.LongTensor(tmp).unsqueeze(-1).unsqueeze(-1) # [128,1,1]
            att_pred = torch.gather(score_map.softmax(1),1,pix_ind.repeat(1,1,100).cuda()).squeeze() # [B*100]
        else: # (2) same location as teacher
            att_q = nn.functional.normalize(torch.gather(score_map.softmax(1),1,pix_ind_k.repeat(1,1,100).cuda()).squeeze(),dim=1) # [B*100] # select local q (attribute original!)
            #att_pred = score_map.softmax(2).mean(1) # local q (attribute global)
            #att_pred = score_map.softmax(1) # all q to all k (attribute all)
            #att_pred = (torch.mm(q,self.cluster_centers.t().cuda())/0.05) # for kmean
        
        #loss_mask = (score_map_k.argmax(2)==labels.unsqueeze(1)).float() # true or false of target cls
        #loss_mask = (score_map_k.argmax(2)==logits_q.argmax(1).unsqueeze(1)).float() # true or false of target cls

        #att_label = torch.gather(score_map.detach().softmax(1),1,pix_ind_k.repeat(1,1,100).cuda()).squeeze() # [B*100]
        #att_loss = nn.MSELoss()(att_pred,att_label) #nn.L1Loss()(att_pred,att_label) #nn.MSELoss()(att_pred,att_label) #nn.BCELoss()(att_pred,att_label) #---!!
        #att_loss = (nn.MSELoss(reduction='none')(att_pred,att_label)*loss_mask.unsqueeze(-1)).mean()
        #att_loss = nn.CrossEntropyLoss()(att_pred,att_label)# kmeans based loss
        if self.cluster_centers is not None:
            att_pred = (torch.mm(att_q,self.cluster_centers.t().cuda())/0.05) #pairwise_distance(att_q,self.cluster_centers.cuda())
            att_label = (torch.mm(att_k.detach(),self.cluster_centers.t().cuda())/0.05) #pairwise_distance(att_k,self.cluster_centers.cuda())
            
            #att_loss = nn.KLDivLoss(reduction='batchmean')(att_pred.log_softmax(1),att_label.softmax(1))# -----(1) kmeans based matching loss
            
            fixmatch_weight = (att_label.detach().softmax(1).max(1)[0]>0.5).float() #---(2)
            att_loss = (nn.CrossEntropyLoss(reduction='none')(att_pred,att_label.argmax(1)) * fixmatch_weight).mean()# (2) kmeans based one-hot loss 
        else:
            att_loss = 0
        #att_loss = nn.MSELoss()(q_encoding,k_encoding) # feat level loss


        # compute logits
        features_local = torch.cat((q_local, k_local, self.queue_local.clone().detach()), dim=0)
        features_global = torch.cat((q_global, k_global, self.queue_global.clone().detach()), dim=0)
        # if self.cluster_centers is not None:
        #     target = torch.cat((att_pred.argmin(1).detach(), att_label.argmin(1).detach(), self.queue_l.clone().detach()), dim=0) #---!!! for clabel    
        # else:
        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0)
        

        #self._dequeue_and_enqueue(k, labels)
        # if self.cluster_centers is not None:#------!!!!
        #     self._dequeue_and_enqueue(k, att_label.argmin(1).detach(), att_k)
        # else:
        self._dequeue_and_enqueue(k_local, k_global, labels, att_k)
        

        
        

        return features_local,features_global, target, logits_q , att_loss #----!!!

    def _inference(self, image):
        q,q_encoding = self.encoder_q(image) #----!!!  w/ encoding output
        q = nn.functional.normalize(q, dim=1)
        encoder_q_logits = self.linear(self.feat_after_avg_q)

        return encoder_q_logits

    def forward(self, im_q, im_k=None, labels=None):
        if self.training:
           return self._train(im_q, im_k, labels) 
        else:
           return self._inference(im_q)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output