from torch import nn

from .MLP import MLP


class LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.fc = MLP(
            in_feats=hidden_dim,
            h_feats=[output_dim],
            acts=[nn.ReLU()],
            dropout=dropout,
        )

    def forward(self, x, nfc=True):
        lstm_out, (h_n, c_n) = self.lstm(x)
        if nfc:
            return h_n[-1]
        out = self.fc(h_n[-1])
        return out
