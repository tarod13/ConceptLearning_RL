import torch

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

# Others
#------------------------
def updateNet(target, source, tau):    
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def get_conv_out(net, shape):
    o = net.conv(torch.zeros(1, *shape))
    return int(np.prod(o.size()))