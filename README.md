# In-Context Learning Dynamics of Linear Attention

## Attention with Merged Key and Query

```bash
python train.py --model mlp --cubic_feat --head 8 --init 1e-6
python train.py --model attnM --head 8 --init 1e-6

```

Figure 1b but with white covariance

```bash
python train.py --model mlp --cubic_feat --head 8 --init 1e-6 --white_cov
```

## Attention with Separate Key and Query

```bash
python train.py --model attnS --head 5 --init 1e-2 --trainset_size 80000 --epoch 10001 --lr 0.02
```
low-rank key and query with input dim D=8 (Jan 14)
```bash
python train.py --model attnS --head 9 --KQ_dim 1 --in_dim 8 --seq_len 32 --init 5e-3 --trainset_size 80000 --epoch 50001 --lr 0.02
```

## In-Context and In-Weight Learning Dynamics

```bash
C="0 0.2 0.4 0.6 0.8 1"
for c in $C; do
  python train.py --model attnM --icl c --head 8 --testset 5000 --init 1e-6 --white_cov --lr 0.0005 --epoch 4001;
done
```
