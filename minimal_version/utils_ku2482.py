# https://github.com/ku2482/sac-discrete.pytorch/blob/42b53f91d30e69b8bc9b415c314fd2c29906856a/code/utils.py#L1
import numpy as np
import torch

def to_batch(state, action, reward, next_state, done, intrinsic_reward, device):
    state = torch.FloatTensor(
        state).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(
        next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    intrinsic_reward = torch.FloatTensor([intrinsic_reward]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done, intrinsic_reward