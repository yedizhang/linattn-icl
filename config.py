import argparse

def config():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument("--epoch", type=int, default=6001, help='number of epochs')
    parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
    parser.add_argument("--seed", type=int, default=10, help='random seed')

    # data set
    parser.add_argument("--icl", type=float, default=1, help='portion of training sequences with a random task vector (icl=1 is a purely icl task)')
    parser.add_argument("--trainset_size", type=int, default=5000, help='number of training samples')
    parser.add_argument("--testset_size", type=int, default=0, help='number of training samples')
    parser.add_argument("--white_cov", action="store_true", help='use a white input token covariance matrix')
    parser.add_argument("--vary_len", action="store_true", help='compute loss on sequences of varying lengths')
    parser.add_argument("--seq_len", type=int, default=32, help='sequence length')
    parser.add_argument("--in_dim", type=int, default=4, help='dimension of x')
    parser.add_argument("--out_dim", type=int, default=1, help='dimension of y')
    parser.add_argument("--cubic_feat", action="store_true", help='map X to cubic features z')
    
    # network
    parser.add_argument("--model", type=str, default='attnM', choices={'attnS', 'attnM', 'mlp'}, help='model type')
    parser.add_argument("--softmax", action="store_true", help='softmax or linear (default) attention')
    parser.add_argument("--rank", type=int, default=1, help='rank of W_K and W_Q matrices')
    parser.add_argument("--head_num", type=int, default=1, help='number of heads')
    parser.add_argument("--init", type=float, default=0.01, help='initialization scale')

    # plotting and saving
    parser.add_argument("--show", action="store_true", help='show matplotlib window if True; otherwise save txt')
    parser.add_argument("--track_value", action="store_true", help='track the dynamics of value weight if True')

    print(parser.parse_args(), '\n')
    return parser