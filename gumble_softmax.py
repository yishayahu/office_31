import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

def sample_gumbel(shape, eps=1e-20,device='cpu'):


    U= torch.FloatTensor(shape).uniform_().to(device)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature,device):
    y = logits + sample_gumbel(logits.size(),device=device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature = 5,device='cpu'):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature,device)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y
