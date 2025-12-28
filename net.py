import torch
import torch.nn as nn


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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=init)


class Attention_Merge(nn.Module):
    # attention with the key and query merged as a single matrix
    def __init__(self, in_dim, out_dim, head_num, init, softmax, vary_len):
        super(Attention_Merge, self).__init__()       
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.softmax = softmax
        self.vary_len = vary_len
        self.embed_dim = in_dim + out_dim

        self.kq = nn.Linear(self.embed_dim, head_num * self.embed_dim, bias=False)
        self.value = nn.Linear(self.embed_dim, head_num * self.embed_dim, bias=False)

        self._init_weights(init)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        B, N, D = x.shape
        H = self.head_num
        KQ = self.kq(x).view(B, N, H, D).transpose(1, 2)    # (B, H, N, D)
        V = self.value(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        attn = torch.matmul(x.unsqueeze(1), KQ.transpose(-1, -2))
        if self.softmax:
            attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.sum(dim=1)  # (B, N, D)
        if self.vary_len:
            out = out / N
        return out

    def _init_weights(self, init):
        for _, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=init)

        for h in range(self.head_num):
            # zero initial weights in merged W_kq
            start_row = h * self.embed_dim
            end_row = start_row + self.in_dim
            nn.init.constant_(self.kq.weight[start_row:end_row, -1], 0)

            # zero initial weights in W_v
            row_idx = (h + 1) * self.embed_dim - 1
            nn.init.constant_(self.value.weight[row_idx, :self.in_dim], 0)


class Attention_Separate(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, rank, init, softmax, vary_len):
        super(Attention_Separate, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.rank = rank
        self.softmax = softmax
        self.vary_len = vary_len
        self.embed_dim = in_dim + out_dim

        self.key = nn.Linear(self.embed_dim, head_num * rank, bias=False)
        self.query = nn.Linear(self.embed_dim, head_num * rank, bias=False)
        self.value = nn.Linear(self.embed_dim, head_num * self.embed_dim, bias=False)

        self._init_weights(init)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        B, N, D = x.shape
        H = self.head_num
        R = self.rank
        K = self.key(x).view(B, N, H, R).transpose(1, 2)    # (B, H, N, R)
        Q = self.query(x).view(B, N, H, R).transpose(1, 2)  # (B, H, N, R)
        V = self.value(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)        
        attn = torch.matmul(Q, K.transpose(-1, -2))         # attention scores: (B, H, N, N)
        if self.softmax:
            attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # multiply value matrix
        out = out.sum(dim=1)         # sum over heads
        if self.vary_len:
            out = out / N
        return out

    def _init_weights(self, init):
        for _, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=init/(layer.weight.numel() ** 0.5))
        
        nn.init.constant_(self.key.weight[:, -1], 0)
        for h in range(1, 1+self.head_num):
            row_idx = h*self.embed_dim - 1
            nn.init.constant_(self.value.weight[row_idx, :self.in_dim], 0)
