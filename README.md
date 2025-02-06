# Training Dynamics of In-Context Learning in Linear Attention

[Paper](https://arxiv.org/abs/2501.16265)

## Setup

Python 3 dependencies:

- pytorch
- numpy
- argparse
- matplotlib

## Dataset

We consider the in-context linear regression task ([Garg et al., 2022](https://arxiv.org/abs/2208.01066); [Zhang et al., 2024](https://www.jmlr.org/papers/v25/23-1042.html)). The input is
$$
\bold X = \begin{bmatrix}
\bold x_1 & \bold x_2  & \cdots & \bold x_N & \bold x_q \\
y_1 & y_2 & \cdots & y_N & 0
\end{bmatrix} \in \mathbb R^{(D+1)\times(N+1)}
$$
where $\bold x \in \mathbb R^D$ and $N$ is the sequence length. The desired output is a linear map of the query input, $y_q = \bold w^\top \bold x_q$. The $y_n$ in context are generated as the same linear map, $y_n = \bold w^\top \bold x_n, n=1,\cdots,N$. Note that the vector $\bold w$, which we call the task vector, varies across different sequences and is independently sampled  from $\mathcal N(\bold 0,\bold I)$.

## Attention with Merged Key and Query

Multi-head linear attention with the key and query matrices merged as a single matrix ${\bold W^K}^\top \bold W^Q = \bold W^{KQ}$, defined as
$$
\textsf{ATTN}_{\text M} (\bold X) = \bold X + \sum_{i=1}^H \frac1N \bold W^V_i \bold X \bold X^\top \bold W^{KQ}_i \bold X
$$
Simulate a loss trajectory of $\textsf{ATTN}_{\text M}$

```bash
python train.py --model attnM --head 8 --init 1e-6 --show
```

When trained on in-context linear regression tasks, the linear attention with merged key and query is equivalent to a 2-layer fully-connected linear networks with a set of cubic features as input
$$
\textsf{ATTN}_{\text M} (\bold X)_{D+1,N+1} = \textsf{MLP} (\bold z)
$$
where the cubic feature is $\bold z = \frac1N \sum_{n=1}^N y_n \bold x_n \bold x_q^\top$. 

If we train $\textsf{MLP} (\bold z)$ with the same dataset and initialization, the loss trajectory will be the same as that of $\textsf{ATTN}_{\text M}$.

```bash
python train.py --model mlp --cubic_feat --head 8 --init 1e-6 --show
```

## Attention with Separate Key and Query

Multi-head linear attention with separate key and query, defined as
$$
\textsf{ATTN}_{\text S} (\bold X) = \bold X + \sum_{i=1}^H \frac1N \bold W^V_i \bold X \bold X^\top {\bold W^K_i}^\top \bold W^Q_i \bold X
$$

### Rank-One Key and Query

Simulate a loss trajectory of rank-one $\textsf{ATTN}_{\text S}$

```bash
python train.py --model attnS --head 5 --init 1e-2 --epoch 10001 --lr 0.02 --show
```
### Low-Rank Key and Query

We vary the rank of the key and query weights and see how the loss trajectories differ from their rank-one counterpart. We set input token dimension $D=8$ and vary the rank (controlled by the `--KQ_dim` parser). The following commands generate the txt file of the loss curve. Add `--show` parser to display the loss curve.

```bash
R="8 4 2 1"
for r in $R; do
  python train.py --model attnS --head 9 --KQ_dim "$r" --in_dim 8 --seq_len 32 --init 5e-3 --trainset_size 80000 --epoch 20001 --lr 0.02
done
```

> [!TIP]
>
> All the commands we provide match what we did to generate the figures in our paper. For just playing with the code, one can use a smaller training dataset, larger initialization, and shorter training epochs. The loss curves may be a little noisier but training can run faster.

## In-Context and In-Weight Learning Dynamics

The `--icl` parser controls the portion of training sequences with randomly sampled task vectors. Its default setting is 1, which means a purely in-context learning task. Setting it below 1 elicits in-weight learning.

```bash
C="0 0.2 0.4 0.6 0.8 1"
for c in $C; do
  python train.py --model attnM --icl c --head 8 --testset 5000 --init 1e-6 --white_cov --lr 0.0005 --epoch 4001;
done
```

## Citation

```
@misc{yedi25icl,
      title={Training Dynamics of In-Context Learning in Linear Attention}, 
      author={Yedi Zhang and Aaditya K. Singh and Peter E. Latham and Andrew Saxe},
      year={2025},
      eprint={2501.16265},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.16265}, 
}
```
