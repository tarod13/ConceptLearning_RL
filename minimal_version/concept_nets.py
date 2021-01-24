import torch
import torch.nn as nn

from policy_nets import softmax_policy_Net, vision_softmax_policy_Net


class concept_Net(nn.Module):
    def __init__(self):
        super().__init__(s_dim, n_concepts, noisy=True)

        self.classifier = softmax_policy_Net(s_dim, n_concepts, noisy=noisy)
        self._n_concepts = n_concepts

    def forward(self, s):
        PS_s, log_PS_s = self.classifier()
        return PS_s, log_PS_s 


class SA_concept_Net(nn.Module):
    def __init__(self):
        super().__init__(s_dim, n_state_concepts, a_dim, n_action_concepts, noisy=True)

        self.state_net = concept_Net(s_dim, n_state_concepts, noisy=noisy)
        self.action_net = concept_Net(a_dim, n_action_concepts, noisy=noisy)

    def forward(self, s, a):
        PS_s, log_PS_s = self.state_net(s)
        PA_a, log_PA_a = self.state_net(a)
        return PS_s, log_PS_s, PA_a, log_PA_a 


class visual_S_concept_Net(nn.Module):
    def __init__(self, s_dim, latent_dim, n_concepts, noisy=True, lr=1e-4):
        super().__init__()

        self.classifier = vision_softmax_policy_Net(s_dim, latent_dim, n_concepts, noisy, lr)
        self._n_concepts = n_concepts

    def forward(self, inner_state, outer_state):
        PS_s, log_PS_s = self.classifier(inner_state, outer_state)
        return PS_s, log_PS_s 