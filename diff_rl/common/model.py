import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_rl.common.helpers import SinusoidalPosEmb

class MLP(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 t_dim=16):

        super(MLP, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = t_dim + state_dim + action_dim # 1 stands for the position for Q-value
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 64),
                                       nn.Tanh())

        self.final_layer = nn.Linear(64, action_dim)

    def forward(self, x, time, state):
        if len(time.shape) > 1:
            time = time.squeeze(1)  # added for shaping t from (batch_size, 1) to (batch_size,)
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1).to(torch.float32)
        x = self.mid_layer(x)

        return self.final_layer(x)

