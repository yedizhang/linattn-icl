import argparse

def config():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument("--epoch", type=int, default=4001, help='number of epochs')
    parser.add_argument("--lr", type=float, default=0.001, help='learning rate')

    # data set
    parser.add_argument("--trainset_size", type=int, default=1000, help='number of training samples')
    parser.add_argument("--testset_size", type=int, default=0, help='number of training samples')
    parser.add_argument("--white_cov", action="store_true", help='use a white input token covariance matrix')
    parser.add_argument("--seq_len", type=int, default=20, help='sequence length')
    parser.add_argument("--in_dim", type=int, default=4, help='dimension of x')
    parser.add_argument("--out_dim", type=int, default=1, help='dimension of y')
    parser.add_argument("--cubic_feat", action="store_true", help='map X to cubic features z')
    
    # network
    parser.add_argument("--model", type=str, default='attn_KQ', choices={'attn', 'attn_KQ', 'transformer', 'mlp'}, help='model type')
    parser.add_argument("--KQ_dim", type=int, default=1, help='dimension of W_K and W_Q matrices')
    parser.add_argument("--head_num", type=int, default=1, help='number of heads')
    parser.add_argument("--init", type=float, default=1e-3, help='initialization scale')

    parser.add_argument("--seed", type=int, default=0, help='random seed')

    print(parser.parse_args(), '\n')
    return parser