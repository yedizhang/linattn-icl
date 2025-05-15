import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net import *
from data import *
from utils import *
from config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def creat_network(args):
    if args.model == 'attnM':
        model = Attention_Merge(args.in_dim, args.out_dim, args.head_num, args.init, args.softmax, args.vary_len).to(device)
    elif args.model == 'attnS':
        model = Attention_Separate(args.in_dim, args.out_dim, args.head_num, args.rank, args.init, args.softmax, args.vary_len).to(device)
    elif args.model == 'mlp':
        model = MLP(args.in_dim, args.out_dim, args.head_num, args.init).to(device)
    print(model)
    return model


def compute_loss(model, data, args):
    if not args.vary_len:
        outputs = model(data["x"])
        if args.model != 'mlp':
            outputs = outputs[:,-1,args.in_dim:]
        loss = nn.MSELoss()(data["y"][:,-1,args.in_dim:], outputs)
    else:
        assert args.model != 'mlp', "MLP with varying context lengths is not implemented."
        loss = 0
        for n in range(2, args.seq_len+1):
            seq = torch.clone(data["x"][:,:n,:])
            seq[:,[-1],args.in_dim:] = 0
            outputs = model(seq)
            loss += nn.MSELoss()(data["y"][:,n-1,args.in_dim:], outputs[:,-1,args.in_dim:])
        loss /= args.seq_len - 1
    return loss


def train(model, data, args):
    results = {'Ls': np.zeros(args.epoch),
               'Eg_iwl': np.zeros(args.epoch),
               'Eg_icl': np.zeros(args.epoch),
               'V': np.zeros((args.epoch,args.head_num))}
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for t in range(args.epoch):
        optimizer.zero_grad()
        loss = compute_loss(model, data, args)
        results['Ls'][t] = loss.item()
        if args.testset_size != 0:
            results['Eg_iwl'][t] = mse(data["y_iwl"], model(data["x_iwl"]), args.in_dim)
            results['Eg_icl'][t] = mse(data["y_icl"], model(data["x_icl"]), args.in_dim)
        if args.track_value:
            W = [param.data.cpu().detach().numpy() for param in model.parameters()]
            for h in range(args.head_num):
                results['V'][t,h] = W[2*args.head_num+h][-1,-1]
        if t % 2000 == 0:
            print(f"Epoch [{t}/{args.epoch}], Loss: {loss.item():.4f}")
            if args.show:
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