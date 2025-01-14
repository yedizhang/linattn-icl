# In-Context Learning Dynamics of Linear Attention

## Attention with Merged Key and Query

```bash
python train.py --model mlp --cubic_feat --head 8 --init 1e-6
python train.py --model attn_KQ --head 8 --init 1e-6

```

Figure 1b but with white covariance

```bash
python train.py --model mlp --cubic_feat --head 8 --init 1e-6 --white_cov
```

## Attention with Separate Key and Query

```bash
python train.py --model attn --head 5 --init 1e-2 --trainset_size 80000 --epoch 10001 --lr 0.02
```

## In-Context and In-Weight Learning Dynamics

```bash
python train.py --model attn_KQ --head 8 --testset 5000 --init 1e-6 --white_cov --in_dim 2 --lr 0.0005 --epoch 4001
```
