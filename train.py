import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import invwishart
from net import *
from utils import *
from config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)


def creat_network(args):
    if args.model == 'attn_KQ':
        model = LinAttention_KQ(args.in_dim, args.out_dim, args.head_num).to(device)
    elif args.model == 'attn':
        model = LinAttention(args.in_dim, args.out_dim, args.head_num, args.KQ_dim).to(device)
    elif args.model == 'transformer':
        model = LinTransformer(args.in_dim, args.out_dim).to(device)
    elif args.model == 'mlp':
        model = MLP(args.in_dim, args.out_dim).to(device)
    print(model)
    return model


def gen_data_(num_samples, seq_len, in_dim, out_dim, cov, w, mode='bursty', cubic_feat=False):
    x = np.random.multivariate_normal(np.zeros(in_dim), cov, size=num_samples*seq_len)
    Sigma_inv = np.linalg.inv(np.cov(x.T))
    eigval, eigvec = np.linalg.eigh(np.linalg.inv(Sigma_inv))
    vis_matrix([Sigma_inv, np.diag(eigval), eigvec], 'Covariance of x')
    x = torch.tensor(np.reshape(x, (num_samples, seq_len, in_dim))).float()
    x = x - torch.mean(x, dim=0, keepdim=True)
    if mode == 'bursty':
        y = torch.matmul(x, w)
    elif mode == 'icl':
        w_ic = torch.normal(0, 1, size=(num_samples, in_dim, out_dim))
        w_ic = whiten(w_ic)
        y = torch.einsum('nli,nio->nlo', x, w_ic)
    elif mode == 'iwl':
        y = torch.randn(num_samples, seq_len, out_dim)
        y[:,-1,:] = torch.matmul(x[:,-1,:], w)
    seq = torch.cat((x, y), dim=2)  # shape [num_samples, seq_len, in_dim+out_dim]
    targets = seq.clone()
    seq[:,[-1],in_dim:] = 0
    if cubic_feat:
        XX = torch.bmm(seq.transpose(1, 2), seq)
        beta_c = XX[:,:in_dim,[-1]]
        # beta_c = XX[:,:,[-1]]
        x_q = seq[:,[-1],:in_dim]
        X_feat = torch.bmm(beta_c, x_q)
        seq = torch.flatten(X_feat, start_dim=1)
        # E_zz = np.cov(seq.T)
        # E_yz = seq.T @ targets[:,-1,in_dim:] / seq.shape[0]
        # vis_matrix(E_zz)
        # vis_matrix(np.array(E_yz))
    return seq.to(device), targets.to(device)


def gen_dataset(args):
    data = {}
    if args.rand_cov:
        w = torch.normal(0, 1, size=(args.in_dim, args.out_dim))
        cov = invwishart.rvs(df=args.in_dim+2, scale=np.eye(args.in_dim))
    else:
        w = torch.ones(args.in_dim, args.out_dim)
        cov = np.eye(args.in_dim)
    cov = cov / np.trace(cov)
    data['x'], data['y'] = gen_data_(args.trainset_size, args.seq_len, args.in_dim, args.out_dim, cov, w, 'icl', args.cubic_feat)
    if args.testset_size != 0:
        data['x_iwl'], data['y_iwl'] = gen_data_(args.testset_size, args.seq_len, args.in_dim, args.out_dim, cov, w, 'iwl', args.cubic_feat)
        data['x_icl'], data['y_icl'] = gen_data_(args.testset_size, args.seq_len, args.in_dim, args.out_dim, cov, w, 'icl', args.cubic_feat)
    print("Trainset shape:", data['x'].shape, data['y'].shape)
    print("Input cov eigvals:", np.linalg.eigvals(cov))
    return data


def train(model, data, args):
    results = {'Ls': np.zeros(args.epoch),
               'Eg_iwl': np.zeros(args.epoch),
               'Eg_icl': np.zeros(args.epoch)}
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for t in range(args.epoch):
        optimizer.zero_grad()
        outputs = model(data["x"])
        if not args.cubic_feat:
            outputs = outputs[:,-1,args.in_dim:]
        loss = nn.MSELoss()(data["y"][:,-1,args.in_dim:], outputs)
        results['Ls'][t] = loss.item()
        if args.testset_size != 0:
            results['Eg_iwl'][t] = mse(data["y_iwl"], model(data["x_iwl"]), args.in_dim)
            results['Eg_icl'][t] = mse(data["y_icl"], model(data["x_icl"]), args.in_dim)
        if t % 2000 == 0:
            print(f"Epoch [{t}/{args.epoch}], Loss: {loss.item():.4f}")
            vis_weight(args, model.parameters(), t)

        loss.backward()
        optimizer.step()
    
    vis_loss(args, results)


if __name__ == "__main__":
    args = config().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = gen_dataset(args)
    model = creat_network(args)
    train(model, data, args)