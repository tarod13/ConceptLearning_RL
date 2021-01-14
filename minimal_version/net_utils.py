import numpy as np
import torch
import torch.nn as nn

from custom_layers import parallel_Linear

#Initializations
#------------------------
def weights_init_rnd(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        try:
            m.bias.data.zero_()
        except:
            pass

def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear): # or isinstance(m, parallel_Linear_simple) or isinstance(m, Linear)
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def weights_init_he(m):
     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0.0)  

# Others
#------------------------
def updateNet(target, source, tau):    
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def get_conv_out(conv_net, shape):
    o = conv_net(torch.zeros(1, *shape))
    return int(np.prod(o.size()))