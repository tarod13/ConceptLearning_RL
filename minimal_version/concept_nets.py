import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Adam

from policy_nets import softmax_policy_Net, vision_softmax_policy_Net


class concept_Net(nn.Module):
    def __init__(self, s_dim, n_concepts, noisy=False):
        super().__init__()

        self.classifier = softmax_policy_Net(s_dim, n_concepts, noisy=noisy)
        self._n_concepts = n_concepts

    def forward(self, s):
        PS_s, log_PS_s = self.classifier(s)
        return PS_s, log_PS_s 


class SA_concept_Net(nn.Module):
    def __init__(self, s_dim, n_state_concepts, a_dim, n_action_concepts, 
        noisy=False, init_log_alpha=1.0):
        super().__init__()

        self.state_net = concept_Net(s_dim, n_state_concepts, noisy)
        self.action_net = concept_Net(a_dim, n_action_concepts, noisy)
        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, init_log_alpha)
        self.alpha_optimizer = Adam([self.log_alpha], lr=3e-4)

    def forward(self, s, a):
        PS_s, log_PS_s = self.state_net(s)
        PA_a, log_PA_a = self.action_net(a)
        return PS_s, log_PS_s, PA_a, log_PA_a 


class visual_S_concept_Net(nn.Module):
    def __init__(self, s_dim, latent_dim, n_concepts, noisy=False, lr=1e-4):
        super().__init__()

        self.classifier = vision_softmax_policy_Net(s_dim, latent_dim, n_concepts, noisy, lr)
        self._n_concepts = n_concepts

    def forward(self, inner_state, outer_state):
        PS_s, log_PS_s = self.classifier(inner_state, outer_state)
        return PS_s, log_PS_s 