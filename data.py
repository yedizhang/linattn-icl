import torch
import numpy as np
from utils import vis_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_cov(args):
    if args.white_cov:
        # covariance matrix of the 'input' token x is white when given white_cov parser
        Lambda = np.eye(args.in_dim)
    else:
        # default covariance matrix of the 'input' token x
        # has eigenvalues lambda_d Lambda proportional to d
        eigval = np.arange(args.in_dim)+1
        Q, _ = np.linalg.qr(np.random.randn(len(eigval), len(eigval)))  # generate a random orthogonal matrix
        Lambda = Q @ np.diag(eigval) @ Q.T
    Lambda = Lambda / np.trace(Lambda)
    return Lambda


def whiten(W):
    # shape of W: (trainset_size, in_dim, out_dim)
    # enforce task vectors w to have zero mean and white covariance
    W = W - torch.mean(W, dim=0, keepdim=True)
    for o in range(W.shape[-1]):
        W_o = W[:,:,o]
        cov_matrix = torch.cov(W_o.T)
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
        whitening_matrix = eigvecs @ torch.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        W[:,:,o] = W_o @ whitening_matrix.T
    return W


def gen_data_(num_samples, seq_len, in_dim, out_dim, cov, w, icl=1, cubic_feat=False):
    x = np.random.multivariate_normal(np.zeros(in_dim), cov, size=num_samples*seq_len)
    Lambda_inv = np.linalg.inv(np.cov(x.T))
    eigval, eigvec = np.linalg.eigh(np.linalg.inv(Lambda_inv))
    # vis_matrix([Lambda_inv, np.diag(eigval), eigvec.T], 'Covariance of x')
    x = torch.tensor(np.reshape(x, (num_samples, seq_len, in_dim))).float()
    x = x - torch.mean(x, dim=0, keepdim=True)

    w_ic = torch.normal(0, 1, size=(num_samples, in_dim, out_dim))
    if icl == 1:
        w_ic = whiten(w_ic)
        y = torch.einsum('nli,nio->nlo', x, w_ic)
    elif icl == 0:
        y = torch.normal(0, 1, size=(num_samples, seq_len, out_dim))
        y[:,-1,:] = torch.matmul(x[:,-1,:], w)
    else:
        icl_num = int(num_samples * icl)
        w_ic[icl_num:] = w
        y = torch.einsum('nli,nio->nlo', x, w_ic)    

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
        # E_zz = np.cov(seq, rowvar=False)
        # E_yz = seq.T @ targets[:,-1,in_dim:] / seq.shape[0]
        # vis_matrix(E_zz)
        # vis_matrix(np.array(E_yz))
        # np.savetxt('linreg.txt', np.linalg.inv(E_zz)@np.array(E_yz))
    return seq.to(device), targets.to(device)


def gen_dataset(args):
    data = {}
    w = torch.normal(0, 1, size=(args.in_dim, args.out_dim))
    w = w * args.in_dim**0.5 / torch.norm(w)
    cov = gen_cov(args)
    data['x'], data['y'] = gen_data_(args.trainset_size, args.seq_len, args.in_dim, args.out_dim, cov, w, args.icl, args.cubic_feat)
    if args.testset_size != 0:
        data['x_iwl'], data['y_iwl'] = gen_data_(args.testset_size, args.seq_len, args.in_dim, args.out_dim, cov, w, 0, args.cubic_feat)
        data['x_icl'], data['y_icl'] = gen_data_(args.testset_size, args.seq_len, args.in_dim, args.out_dim, cov, w, 1, args.cubic_feat)
    print("Trainset shape:", data['x'].shape, data['y'].shape)
    print("Eigval of Lambda:", np.linalg.eigvals(cov))
    return data