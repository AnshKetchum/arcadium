import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKVLayer(nn.Module):
    def __init__(self, n_embd=1024):
        super().__init__()

        # ---- LayerNorms ----
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # ---- Time Mixing parameters ----
        self.time_decay = nn.Parameter(torch.zeros(n_embd))
        self.time_bonus = nn.Parameter(torch.zeros(n_embd))
        self.time_mix_k = nn.Parameter(torch.zeros(n_embd))
        self.time_mix_v = nn.Parameter(torch.zeros(n_embd))
        self.time_mix_r = nn.Parameter(torch.zeros(n_embd))

        self.Wk = nn.Linear(n_embd, n_embd, bias=False)
        self.Wv = nn.Linear(n_embd, n_embd, bias=False)
        self.Wr = nn.Linear(n_embd, n_embd, bias=False)
        self.Wout = nn.Linear(n_embd, n_embd, bias=False)

        # ---- Channel Mixing parameters ----
        self.channel_mix_k = nn.Parameter(torch.zeros(n_embd))
        self.channel_mix_r = nn.Parameter(torch.zeros(n_embd))

        self.Wk_ffn = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.Wr_ffn = nn.Linear(n_embd, n_embd, bias=False)
        self.Wv_ffn = nn.Linear(4 * n_embd, n_embd, bias=False)

    def time_mixing(self, x, last_x, last_num, last_den):
        k = self.Wk(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        v = self.Wv(x * self.time_mix_v + last_x * (1 - self.time_mix_v))
        r = self.Wr(x * self.time_mix_r + last_x * (1 - self.time_mix_r))

        wkv = (last_num + torch.exp(self.time_bonus + k) * v) / \
              (last_den + torch.exp(self.time_bonus + k))
        rwkv = torch.sigmoid(r) * wkv
        out = self.Wout(rwkv)

        num = torch.exp(-torch.exp(self.time_decay)) * last_num + torch.exp(k) * v
        den = torch.exp(-torch.exp(self.time_decay)) * last_den + torch.exp(k)

        return out, (x, num, den)

    def channel_mixing(self, x, last_x):
        k = self.Wk_ffn(x * self.channel_mix_k + last_x * (1 - self.channel_mix_k))
        k = F.relu(k) ** 2  # squared ReLU

        r = self.Wr_ffn(x * self.channel_mix_r + last_x * (1 - self.channel_mix_r))
        r = torch.sigmoid(r)

        vk = self.Wv_ffn(k)
        return r * vk, x

    def forward(self, x, state):
        # state: (4, n_embd)
        # [0]=last_x_time, [1]=last_num, [2]=last_den, [3]=last_x_channel
        last_x_time, last_num, last_den, last_x_channel = state

        x_ = self.ln1(x)
        dx, (new_x, new_num, new_den) = self.time_mixing(x_, last_x_time, last_num, last_den)
        x = x + dx

        x_ = self.ln2(x)
        dx, new_x_channel = self.channel_mixing(x_, last_x_channel)
        x = x + dx

        new_state = torch.stack([new_x, new_num, new_den, new_x_channel])
        return x, new_state
