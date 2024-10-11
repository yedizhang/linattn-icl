import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from net import MLP, LinTransformer, LinAttention
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


def gen_data_(num_samples, seq_len, input_dim, output_dim, w, mode='bursty'):
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
    return seq, targets


def gen_dataset(num_samples, seq_len, input_dim, output_dim):
    # w = torch.normal(0, 1, size=(input_dim, output_dim))
    w = torch.ones(input_dim, output_dim)
    x, y = gen_data_(num_samples, seq_len, input_dim, output_dim, w, 'iwl')
    x_iwl, y_iwl = gen_data_(num_samples, seq_len, input_dim, output_dim, w, 'iwl')
    x_icl, y_icl = gen_data_(num_samples, seq_len, input_dim, output_dim, w, 'icl')
    print(x.shape, y.shape)
    return {"x": x.to(device),
            "y": y.to(device),
            "x_iwl": x_iwl.to(device),
            "y_iwl": y_iwl.to(device),
            "x_icl": x_icl.to(device),
            "y_icl": y_icl.to(device)}


def mse(y, y_hat, input_dim):
    return F.mse_loss(y[:,[-1],input_dim:], y_hat[:,[-1],input_dim:]).cpu().detach().numpy()


def vis_weight(params):
    W = [param.data.cpu().detach().numpy() for param in params]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    KQ = W[1].T @ W[0]
    thresh_KQ = np.max(np.abs(KQ))
    thresh_V = np.max(np.abs(W[2]))
    im0 = ax[0].imshow(W[2], cmap='coolwarm', vmin=-thresh_V, vmax=thresh_V)
    im1 = ax[1].imshow(KQ, cmap='coolwarm', vmin=-thresh_KQ, vmax=thresh_KQ)
    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])
    plt.show()


def vis_loss(results):
    plt.figure(figsize=(4, 3))
    plt.plot(results['Eg_iwl'], c='b')
    plt.plot(results['Eg_icl'], c='r')
    plt.plot(results['Ls'], c='k')
    plt.show()


def train(model, data, input_dim, num_epochs, learning_rate):
    results = {'Ls': np.zeros(num_epochs),
               'Eg_iwl': np.zeros(num_epochs),
               'Eg_icl': np.zeros(num_epochs)}
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(data["x"])
        loss = nn.MSELoss()(outputs[:,[-1],input_dim:], data["y"][:,[-1],input_dim:])
        results['Ls'][t] = loss.item()
        results['Eg_iwl'][t] = mse(data["y_iwl"], model(data["x_iwl"]), input_dim)
        results['Eg_icl'][t] = mse(data["y_icl"], model(data["x_icl"]), input_dim)
        if t % 2000 == 0:
            print(f"Epoch [{t}/{num_epochs}], Loss: {loss.item():.4f}")
            vis_weight(model.parameters())

        loss.backward()
        optimizer.step()
    
    vis_loss(results)


if __name__ == "__main__":
    num_samples = 1000
    seq_len = 20
    input_dim = 5
    output_dim = 1
    learning_rate = 0.002
    num_epochs = 5000

    data = gen_dataset(num_samples, seq_len, input_dim, output_dim)
    model = LinAttention(input_dim, output_dim).to(device)
    print(model)
    train(model, data, input_dim, num_epochs, learning_rate)