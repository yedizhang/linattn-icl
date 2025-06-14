import torch
import torch.nn as nn


def rand_weight(init, size, seed=7):
    torch.manual_seed(seed)
    return torch.randn(size) * init


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid, init):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim**2, hid, bias=False)
        self.fc2 = nn.Linear(hid, out_dim, bias=False)
        self._init_weights(init)

    def forward(self, x):
        fc = self.fc1(x)
        fc = self.fc2(fc)
        return fc

    def _init_weights(self, init):
        # W1 = rand_weight(init, (8, 16))
        # W2 = rand_weight(init, (1, 8))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=init)
                # with torch.no_grad():
                #     if m.weight.shape[0] == 1:
                #         m.weight.copy_(W2)
                #     else:
                #         m.weight.copy_(W1)


class Attention_Merge(nn.Module):
    # linear attention with the key and query merged as a single matrix
    def __init__(self, in_dim, out_dim, head_num, init, softmax, vary_len):
        super(Attention_Merge, self).__init__()       
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
        self.softmax = softmax
        self.vary_len = vary_len
        self._init_weights(init)

    def forward(self, x):
        # x: (batch_size, seq_len, in_dim+out_dim)
        multihead_output = []
        for i in range(self.head_num):
            kq = self.KQ[i](x)  # (batch_size, seq_len, in_dim+out_dim)
            V = self.value[i](x)  # (batch_size, seq_len, in_dim+out_dim)
            attention_scores = torch.bmm(x, kq.transpose(1, 2))  # (batch_size, seq_len, seq_len)
            if self.softmax:
                attention_scores = torch.softmax(attention_scores, dim=-1)
            head_output = torch.bmm(attention_scores, V)
            multihead_output.append(head_output)
        output = sum(multihead_output)  # sum outputs from all heads (batch_size, seq_len, head_dim)
        if self.vary_len:
            output /= x.shape[1]  # N = x.shape[1] is the context length
        return output

    def _init_weights(self, init):
        # W1 = rand_weight(init, (8, 16))
        # W2 = rand_weight(init, (1, 8))
        # i, j = 0, 0
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=init)
                if name.startswith("KQ"):
                    # with torch.no_grad():
                    #     layer.weight[:4, :4] = W1[i].reshape(4,4)
                    # i += 1
                    nn.init.constant_(layer.weight[:self.in_dim,-1], 0)
                if name.startswith("value"):
                    # with torch.no_grad():
                    #     layer.weight[-1, -1] = W2[:,j]
                    # j += 1
                    nn.init.constant_(layer.weight[-1,:self.in_dim], 0)


class Attention_Separate(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, rank, init, softmax, vary_len):
        super(Attention_Separate, self).__init__()
        self.key = nn.ModuleList([
            nn.Linear(in_dim+out_dim, rank, bias=False)
            for _ in range(head_num)
        ])
        self.query = nn.ModuleList([
            nn.Linear(in_dim+out_dim, rank, bias=False)
            for _ in range(head_num)
        ])
        self.value = nn.ModuleList([
            nn.Linear(in_dim+out_dim, in_dim+out_dim, bias=False)
            for _ in range(head_num)
        ])
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.rank = rank
        self.softmax = softmax
        self.vary_len = vary_len
        self._init_weights(init)

    def forward(self, x):
        # x: (num_samples, seq_len, in_dim)
        multihead_output = []
        for i in range(self.head_num):
            K = self.key[i](x)    # (batch_size, seq_len, rank)
            Q = self.query[i](x)  # (batch_size, seq_len, rank)
            V = self.value[i](x)  # (batch_size, seq_len, in_dim+out_dim)
            attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
            if self.softmax:
                attention_scores = torch.softmax(attention_scores, dim=-1)
            head_output = torch.bmm(attention_scores, V)
            multihead_output.append(head_output)
        output = sum(multihead_output)
        if self.vary_len:
            output /= x.shape[1]
        return output

    def _init_weights(self, init):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=init/(torch.numel(layer.weight)**0.5))
                if name.startswith("key"):
                    nn.init.constant_(layer.weight[:,-1], 0)
                if name.startswith("value"):
                    nn.init.normal_(layer.weight, mean=0, std=init)
                    nn.init.constant_(layer.weight[-1,:self.in_dim], 0)
