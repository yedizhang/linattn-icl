import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.rc('font', family="Arial")
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = '14'


def gen_cov(args):
    if args.white_cov:
        Lambda = np.eye(args.in_dim)
    else:
        eigval = np.arange(args.in_dim)+1
        Q, _ = np.linalg.qr(np.random.randn(len(eigval), len(eigval)))  # generate a random orthogonal matrix
        Lambda = Q @ np.diag(eigval) @ Q.T
    Lambda = Lambda / np.trace(Lambda)
    return Lambda


def whiten(X):
    X = X - torch.mean(X, dim=0, keepdim=True)
    for o in range(X.shape[-1]):
        X_o = X[:,:,o]
        cov_matrix = torch.cov(X_o.T)
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
        whitening_matrix = eigvecs @ torch.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        X[:,:,o] = X_o @ whitening_matrix.T
    return X


def mse(y, y_hat, in_dim):
    if y.dim() == 3:
        y = y[:,-1,in_dim:]
    if y_hat.dim() == 3:
        y_hat = y_hat[:,-1,in_dim:]
    return F.mse_loss(y, y_hat).cpu().detach().numpy()


def vis_matrix(M, t=0):
    if not isinstance(M, list):
        thresh = np.max(np.abs(M))
        plt.imshow(M, cmap='RdBu', vmin=-thresh, vmax=thresh)
        plt.colorbar()
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
    else:
        num = len(M)
        fig, ax = plt.subplots(nrows=1, ncols=num)
        for n in range(num):
            mat = M[n]
            thresh = np.max(np.abs(mat))
            im = ax[n].imshow(mat, cmap='RdBu', vmin=-thresh, vmax=thresh)
            fig.colorbar(im, ax=ax[n])
            ax[n].set_xticks([])
            ax[n].set_yticks([])
        fig.suptitle(t)
    plt.tight_layout()
    plt.show()

def vis_weight(args, params, t=0):
    W = [param.data.cpu().detach().numpy() for param in params]
    if args.model == 'attn':
        if args.head_num == 1:
            KQ = W[0].T @ W[args.head_num]
            V = W[-1]
        else:
            V = np.sum(W[2*args.head_num:], axis=0)
            KQ = np.array(W[:args.head_num]).squeeze().T @ np.array(W[args.head_num:2*args.head_num]).squeeze()
        vis_matrix([np.array(W[:args.head_num]).squeeze(), np.array(W[args.head_num:2*args.head_num]).squeeze()], t)
    elif args.model == 'attn_KQ':
        KQ = W[0].T
        V = W[-1]
    elif args.model == 'mlp':
        KQ = W[0].reshape(args.in_dim, args.in_dim)
        V = W[1]
    vis_matrix([V,KQ], t)


def vis_loss(args, results):
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.figure(figsize=(4, 3))
    if results['Eg_iwl'][0] != 0:
        plt.plot(results['Eg_iwl'], c='b')
        plt.plot(results['Eg_icl'], c='r')
    plt.plot(results['Ls'], c='k', lw=2)
    plt.xlim([0, len(results['Ls'])])
    plt.ylim([0, np.max(results['Ls'])+0.1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout(pad=0.5)
    plt.show()
    # np.savetxt(f'{args.model}_head{args.head_num}_KQdim{args.KQ_dim}_P{args.trainset_size}_N{args.seq_len}_seed{args.seed}.txt', results['Ls'])