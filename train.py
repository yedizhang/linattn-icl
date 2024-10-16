import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from net import *
from config import *
plt.rc('font', family="Arial")
plt.rcParams['font.size'] = '14'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def creat_network(args):
    if args.model == 'attn':
        model = LinAttention(args.in_dim, args.out_dim).to(device)
    elif args.model == 'attn_KQ':
        model = LinAttention_KQ(args.in_dim, args.out_dim).to(device)
    elif args.model == 'transformer':
        model = LinTransformer(args.in_dim, args.out_dim).to(device)
    elif args.model == 'mlp':
        model = MLP(args.in_dim, args.out_dim).to(device)
    return model


def gen_data_(num_samples, seq_len, input_dim, output_dim, w, mode='bursty', cubic_feat=False):
    x = torch.normal(0, 1, size=(num_samples, seq_len, input_dim))
    if mode in ['bursty', 'icl']:
        if mode == 'bursty':
            y = torch.matmul(x, w)
        elif mode == 'icl':
            w_ic = torch.randn(num_samples, input_dim, output_dim)
            y = torch.einsum('nli,nio->nlo', x, w_ic)
    elif mode == 'iwl':
        y = torch.randn(num_samples, seq_len, output_dim)
        y[:,-1,:] = torch.matmul(x[:,-1,:], w)
    seq = torch.cat((x, y), dim=2)  # shape [num_samples, seq_len, input_dim+output_dim]
    targets = seq.clone()
    seq[:,[-1],input_dim:] = 0
    if cubic_feat:
        XX = torch.bmm(seq.transpose(1, 2), seq)
        beta_c = XX[:,:input_dim,[-1]]
        x_q = seq[:,[-1],:input_dim]
        X_feat = torch.bmm(beta_c, x_q)
        seq = torch.flatten(X_feat, start_dim=1)
    return seq, targets


def gen_dataset(args):
    w = torch.normal(0, 1, size=(args.in_dim, args.out_dim))
    # w = torch.ones(args.in_dim, args.out_dim)
    x, y = gen_data_(args.trainset_size, args.seq_len, args.in_dim, args.out_dim, w, 'icl', args.cubic_feat)
    x_iwl, y_iwl = gen_data_(args.testset_size, args.seq_len, args.in_dim, args.out_dim, w, 'iwl', args.cubic_feat)
    x_icl, y_icl = gen_data_(args.testset_size, args.seq_len, args.in_dim, args.out_dim, w, 'icl', args.cubic_feat)
    print("Trainset shape:", x.shape, y.shape)
    return {"x": x.to(device),
            "y": y.to(device),
            "x_iwl": x_iwl.to(device),
            "y_iwl": y_iwl.to(device),
            "x_icl": x_icl.to(device),
            "y_icl": y_icl.to(device)}


def mse(y, y_hat, input_dim):
    return F.mse_loss(y[:,[-1],input_dim:], y_hat[:,[-1],input_dim:]).cpu().detach().numpy()


def vis_weight(args, params):
    W = [param.data.cpu().detach().numpy() for param in params]
    if args.model == 'attn':
        KQ = W[1].T @ W[0]
        V = W[2]
    elif args.model == 'attn_KQ':
        KQ = W[0].T
        V = W[1]
    elif args.model == 'mlp':
        KQ = W[0].reshape(args.in_dim, args.in_dim)
        V = W[1]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    thresh_KQ = np.max(np.abs(KQ))
    thresh_V = np.max(np.abs(V))
    im0 = ax[0].imshow(V, cmap='RdBu', vmin=-thresh_V, vmax=thresh_V)
    im1 = ax[1].imshow(KQ, cmap='RdBu', vmin=-thresh_KQ, vmax=thresh_KQ)
    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.show()


def vis_loss(results):
    plt.figure(figsize=(4, 3))
    plt.plot(results['Eg_iwl'], c='b')
    plt.plot(results['Eg_icl'], c='r')
    plt.plot(results['Ls'], c='k')
    plt.show()


def train(model, data, args):
    results = {'Ls': np.zeros(args.epoch),
               'Eg_iwl': np.zeros(args.epoch),
               'Eg_icl': np.zeros(args.epoch)}
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for t in range(args.epoch):
        optimizer.zero_grad()
        outputs = model(data["x"])
        loss = nn.MSELoss()(outputs[:,[-1],args.in_dim:], data["y"][:,[-1],args.in_dim:])
        results['Ls'][t] = loss.item()
        results['Eg_iwl'][t] = mse(data["y_iwl"], model(data["x_iwl"]), args.in_dim)
        results['Eg_icl'][t] = mse(data["y_icl"], model(data["x_icl"]), args.in_dim)
        if t % 2000 == 0:
            print(f"Epoch [{t}/{args.epoch}], Loss: {loss.item():.4f}")
            vis_weight(args, model.parameters())

        loss.backward()
        optimizer.step()
    
    vis_loss(results)


if __name__ == "__main__":
    args = config().parse_args()
    data = gen_dataset(args)
    model = creat_network(args)
    print(model)
    train(model, data, args)