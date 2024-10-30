import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim**2, hid, bias=False)
        self.fc2 = nn.Linear(hid, out_dim, bias=False)
        self._init_weights()

    def forward(self, x):
        fc = self.fc1(x)
        fc = self.fc2(fc)
        return fc

    def _init_weights(self, gamma=1e-4):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=gamma)


class LinTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, KQ_dim=64):
        super(LinTransformer, self).__init__()
        self.query = nn.Linear(in_dim+out_dim, KQ_dim, bias=False)
        self.key = nn.Linear(in_dim+out_dim, KQ_dim, bias=False)
        self.value = nn.Linear(in_dim+out_dim, in_dim+out_dim, bias=False)

        self.fc1 = nn.Linear(in_dim+out_dim, KQ_dim, bias=False)
        self.fc2 = nn.Linear(KQ_dim, in_dim+out_dim, bias=False)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self._init_weights()

    def forward(self, x):
        # x: (num_samples, seq_len, in_dim + out_dim)
        Q = self.query(x)  # (batch_size, seq_len, KQ_dim)
        K = self.key(x)    # (batch_size, seq_len, KQ_dim)
        V = self.value(x)

        # Compute attention (without softmax)
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention = torch.bmm(attention_scores, V)  # (batch_size, seq_len, in_dim + out_dim)

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
    def __init__(self, in_dim, out_dim, head_num=1, KQ_dim=64):
        super(LinAttention, self).__init__()
        self.key = nn.ModuleList([
            nn.Linear(in_dim+out_dim, KQ_dim, bias=False)
            for _ in range(head_num)
        ])
        self.query = nn.ModuleList([
            nn.Linear(in_dim+out_dim, KQ_dim, bias=False)
            for _ in range(head_num)
        ])
        self.value = nn.ModuleList([
            nn.Linear(in_dim+out_dim, in_dim+out_dim, bias=False)
            for _ in range(head_num)
        ])
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.KQ_dim = KQ_dim
        self._init_weights()

    def forward(self, x):
        # x: (num_samples, seq_len, in_dim)
        multihead_output = []
        for i in range(self.head_num):
            K = self.key[i](x)    # (batch_size, seq_len, KQ_dim)
            Q = self.query[i](x)  # (batch_size, seq_len, KQ_dim)
            V = self.value[i](x)  # (batch_size, seq_len, in_dim+out_dim)
            attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
            head_output = torch.bmm(attention_scores, V)
            multihead_output.append(head_output)
        output = sum(multihead_output)
        return output

    def _init_weights(self, gamma=1e-4):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.shape[0] == self.KQ_dim:
                    print(m.weight.shape)
                    nn.init.normal_(m.weight, mean=0, std=(gamma/(self.in_dim+self.out_dim))**0.5)
                else:
                    nn.init.normal_(m.weight, mean=0, std=gamma)


class LinAttention_KQ(nn.Module):
    def __init__(self, in_dim, out_dim, head_num=1):
        super(LinAttention_KQ, self).__init__()       
        self.KQ = nn.ModuleList([
            nn.Linear(in_dim+out_dim, in_dim+out_dim, bias=False)
            for _ in range(head_num)
        ])
        self.value = nn.ModuleList([
            nn.Linear(in_dim+out_dim, in_dim+out_dim, bias=False)
            for _ in range(head_num)
        ])
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self._init_weights()

    def forward(self, x):
        # x: (batch_size, seq_len, in_dim+out_dim)        
        multihead_output = []
        for i in range(self.head_num):
            kq = self.KQ[i](x)  # (batch_size, seq_len, in_dim+out_dim)
            V = self.value[i](x)  # (batch_size, seq_len, in_dim+out_dim)
            attention_scores = torch.bmm(x, kq.transpose(1, 2))  # (batch_size, seq_len, seq_len)
            head_output = torch.bmm(attention_scores, V)
            multihead_output.append(head_output)
        output = sum(multihead_output)  # sum outputs from all heads (batch_size, seq_len, head_dim)
        return output

    def _init_weights(self, gamma=1e-6):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=gamma)