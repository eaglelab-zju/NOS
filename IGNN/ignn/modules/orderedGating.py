import torch
import torch.nn.functional as F
from torch import nn


class ONGNNConv(nn.Module):
    def __init__(
        self,
        tm_net,
        tm_norm,
        simple_gating,
        tm,
        diff_or,
        repeats,
    ) -> None:
        super(ONGNNConv, self).__init__()
        self.tm_net = tm_net
        self.tm_norm = tm_norm
        self.simple_gating = simple_gating
        self.tm = tm
        self.diff_or = diff_or
        self.repeats = repeats

    def forward(self, x, m, last_tm_signal):
        if self.tm == True:
            if self.simple_gating == True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                if self.diff_or == True:
                    tm_signal_raw = last_tm_signal + (1 - last_tm_signal) * tm_signal_raw
            tm_signal = tm_signal_raw.repeat_interleave(repeats=self.repeats, dim=1)
            out = x * tm_signal + m * (1 - tm_signal)
        else:
            out = m
            tm_signal_raw = last_tm_signal

        out = self.tm_norm(out)

        return out, tm_signal_raw
