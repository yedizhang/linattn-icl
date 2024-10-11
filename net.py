import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hid=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim+output_dim, hid, bias=False)
        self.fc2 = nn.Linear(hid, input_dim+output_dim, bias=False)
        self._init_weights()

    def forward(self, x):
        fc = self.fc1(x)
        fc = self.fc2(fc)
        return fc

    def _init_weights(self, gamma=1e-3):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=gamma)


class LinTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hid=128):
        super(LinTransformer, self).__init__()
        self.query = nn.Linear(input_dim+output_dim, hid, bias=False)
        self.key = nn.Linear(input_dim+output_dim, hid, bias=False)
        self.value = nn.Linear(input_dim+output_dim, input_dim+output_dim, bias=False)

        self.fc1 = nn.Linear(input_dim+output_dim, hid, bias=False)
        self.fc2 = nn.Linear(hid, input_dim+output_dim, bias=False)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self._init_weights()

    def forward(self, x):
        # x: (num_samples, seq_len, input_dim + output_dim)
        Q = self.query(x)  # (batch_size, seq_len, hid)
        K = self.key(x)    # (batch_size, seq_len, hid)
        V = self.value(x)

        # Compute attention (without softmax)
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention = torch.bmm(attention_scores, V)  # (batch_size, seq_len, input_dim + output_dim)

        x_skip = x + attention
        fc = self.fc1(x_skip)
        fc = self.fc2(fc)

        return fc


    def _init_weights(self, gamma=1e-3):
        l = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                l += 1
                nn.init.normal_(m.weight, mean=0, std=gamma)
                print(l)
                if l == 4 or l == 5:
                    print(m.weight.shape)
                    nn.init.normal_(m.weight, mean=0, std=1e-12)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=gamma)


class LinAttention(nn.Module):
    def __init__(self, input_dim, output_dim, hid=64):
        super(LinAttention, self).__init__()
        self.query = nn.Linear(input_dim+output_dim, hid, bias=False)
        self.key = nn.Linear(input_dim+output_dim, hid, bias=False)
        self.value = nn.Linear(input_dim+output_dim, input_dim+output_dim, bias=False)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid = hid

        self._init_weights()

    def forward(self, x):
        # x: (num_samples, seq_len, input_dim)      
        Q = self.query(x)  # (batch_size, seq_len, output_dim)
        K = self.key(x)    # (batch_size, seq_len, output_dim)
        V = self.value(x)  # (batch_size, seq_len, output_dim)
        
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        # attention_scores = F.softmax(attention_scores, dim=-1)

        output = torch.bmm(attention_scores, V)  # (batch_size, seq_len, output_dim)

        return output


    def _init_weights(self, gamma=1e-6):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.shape[0] == self.hid:
                    print(m.weight.shape)
                    nn.init.normal_(m.weight, mean=0, std=(gamma/(self.input_dim+self.output_dim))**0.5)
                else:
                    nn.init.normal_(m.weight, mean=0, std=gamma)