import torch
import torch.nn as nn

from custom_layers import Linear_noisy
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


# Simple convolutional net that outputs a feature vector
class vision_Net(nn.Module):
    def __init__(self, latent_dim=256, input_channels=3, height=84, width=168, noisy=True):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.apply(weights_init_he)

        conv_out_size = get_conv_out(self.conv, [input_channels, height, width])
        if noisy:
            self.fc = Linear_noisy(conv_out_size, latent_dim)
            torch.nn.init.orthogonal_(self.fc.mean_weight, 0.01)
            self.fc.mean_bias.data.zero_()
        else:
            self.fc = nn.Linear(conv_out_size, latent_dim)
            torch.nn.init.orthogonal_(self.fc.weight, 0.01)
            self.fc.bias.data.zero_()
        
    def forward(self, x):
        conv_feat = self.conv(x)
        conv_feat = conv_feat.view(x.shape[0], -1) # Squeeze dimensions
        feat = self.fc(conv_feat)
        return feat