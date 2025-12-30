import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.rc('font', family="Arial")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = '14'


def mse(y, y_hat, in_dim):
    if y.dim() == 3:
        y = y[:,-1,in_dim:]
    if y_hat.dim() == 3:
        y_hat = y_hat[:,-1,in_dim:]
    return F.mse_loss(y, y_hat).cpu().detach().numpy()


def vis_matrix(M, t=0):
    if not isinstance(M, list):
        plt.figure(figsize=(3, 3))
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
    # plt.savefig(f't{t:05d}.jpg', dpi=300)
    # plt.close()
    plt.show()


def vis_weight(args, params, t=0):
    W = [param.data.cpu().detach().numpy() for param in params]
    if args.model == 'attnM':
        W[0] = np.reshape(W[0], (args.head_num, -1)).T
        W[1] = np.reshape(W[1], (args.head_num, -1)).T
    vis_matrix(W, f'Epoch {t}')
    # np.savetxt(f'{args.model}_H{args.head_num}_KQ_t{t:04d}.txt', W[0])


def vis_loss(args, results):
    if args.show:
        import matplotlib 
        cmap = matplotlib.colormaps['RdBu']
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.figure(figsize=(4, 3))
        plt.plot(results['Ls'], c='k', lw=2, label='train')
        if results['Eg_iwl'][0] != 0:
            plt.plot(results['Eg_iwl'], c=cmap(0.85), lw=2, label='test IW')
            plt.plot(results['Eg_icl'], c=cmap(0.15), lw=2, label='test IC')
            plt.legend(frameon=False)
        if results['V'][0,0] != 0:
            for h in range(args.head_num):
                plt.plot(results['V'][:,h])
        plt.xlim([0, len(results['Ls'])])
        plt.ylim([0, np.max(results['Ls'])+0.1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout(pad=0.5)
        plt.show()
    else:
        file_id = f'{args.model}_head{args.head_num}_D{args.in_dim}_R{args.rank}_P{args.trainset_size}_N{args.seq_len}_seed{args.seed}'
        if results['Eg_iwl'][0] != 0:
            np.savetxt(f'{file_id}_icl{args.icl}.txt', np.stack((results['Ls'], results['Eg_iwl'], results['Eg_icl']), axis=1))
        elif results['V'][0,0] != 0:
            np.savetxt(f'{file_id}_value.txt', np.hstack((results['Ls'][:,np.newaxis], results['V'])))
        else:
            np.savetxt(f'{file_id}.txt', results['Ls'])