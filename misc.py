from __future__ import print_function, division
import torch.nn as nn
import torch

class OriginLossfuc(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        ap_sim = cal_cosine_similarity(a, p)
        an_sim = cal_cosine_similarity(a, n)
        ap_sim = torch.diag(ap_sim)
        an_sim = torch.diag(an_sim)
        zero = torch.zeros_like(ap_sim)
        res = torch.maximum(zero, an_sim - ap_sim + self.margin)
        return torch.mean(res)
class MemoryBank():
    def __init__(self,num_class,emb_size,device):
        self.all_sum = torch.zeros_like(torch.Tensor(num_class,emb_size)).to(device)
        self.count = torch.zeros_like(torch.Tensor(num_class)).to(device)
        self.device = device
        self.num_class =num_class
        self.emb_size =emb_size
        self.EM = torch.zeros_like(torch.Tensor(num_class,emb_size)).to(device)
    def update(self,outputs,target):
        for o,t in zip(outputs,target):
            self.all_sum[t] += o
            self.count[t] += 1


    def reset(self,num_class,emb_size,device):
        self.device = device
        self.num_class = num_class
        self.emb_size = emb_size
        self.all_sum = torch.zeros_like(torch.Tensor(num_class, emb_size)).to(device)
        self.count = torch.zeros_like(torch.Tensor(num_class)).to(device)
        self.EM = torch.zeros_like(torch.Tensor(num_class, emb_size)).to(device)


    def getEM(self):
        count = self.count.unsqueeze(1).repeat(1,self.emb_size)
        self.EM = self.all_sum / count
        return self.EM

def cal_cosine_similarity(t_a,t_b):
    t_a,t_b = t_a.float(),t_b.float()
    temp = torch.norm(t_b, dim=-1, keepdim=True)
    t_b = t_b / temp

    temp = torch.norm(t_a,dim=1,keepdim=True)
    t_a = t_a / temp

    sims = torch.matmul(t_a,t_b.t())
    return sims


class AverageMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self,val,count):
        self.val = val
        self.sum += val
        self.count += count
        self.avg = self.sum / self.count



if __name__ == '__main__':
    #做一些memory_bank的测试
    mb = MemoryBank(2,3,'cpu')
    a = torch.tensor([[1,2,3],[3,4,5],[4,5,6]])
    b = torch.tensor([0,1,0])
    mb.update(a,b)
    c= mb.getEM()
    d = torch.tensor([[1,2,3],[3,4,5],[4,5,6]])
    e = cal_cosine_similarity(d,c)
    print()