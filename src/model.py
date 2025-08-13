import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMAutoencoder, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, input_size)

        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = "cpu"

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        return (torch.zeros(1, batch_size, self.hidden_size).to(self._device),
                torch.zeros(1, batch_size, self.hidden_size).to(self._device))

    def forward(self, x):
        output, (hidden, cell) = self.encoder(x)
        output = self.linear(output)
        return output