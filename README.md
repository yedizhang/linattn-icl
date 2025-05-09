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
```math
\mathbf X = \begin{bmatrix}
\mathbf x_1 & \mathbf x_2  & \cdots & \mathbf x_N & \mathbf x_q \\
y_1 & y_2 & \cdots & y_N & 0
\end{bmatrix} \in \mathbb{R} ^{(D+1)\times(N+1)}
```
where $\mathbf x \in \mathbb R^D$ and $N$ is the sequence length. The desired output is a linear map of the query input, $y_q = \mathbf w^\top \mathbf x_q$. The $y_n$ in context are generated as the same linear map, $y_n = \mathbf w^\top \mathbf x_n, n=1,\cdots,N$. Note that the vector $\mathbf w$, which we call the task vector, varies across different sequences and is independently sampled  from $\mathcal N(\mathbf 0,\mathbf I)$.

## Attention with Merged Key and Query

Multi-head linear attention with the key and query matrices merged as a single matrix ${\mathbf W^K}^\top \mathbf W^Q = \mathbf W^{KQ}$, defined as
```math
\textsf{ATTN}_{\text M} (\mathbf X) = \mathbf X + \sum_{i=1}^H \frac1N \mathbf W^V_i \mathbf X \mathbf X^\top \mathbf W^{KQ}_i \mathbf X
```
Simulate a loss trajectory of $\textsf{ATTN}_{\text M}$

```bash
python train.py --model attnM --head 8 --init 1e-6 --show
```

When trained on in-context linear regression tasks, the linear attention with merged key and query is equivalent to a 2-layer fully-connected linear networks with a set of cubic features as input
```math
\textsf{ATTN}_{\text M} (\mathbf X)_{D+1,N+1} = \textsf{MLP} (\mathbf z)
```
where the cubic feature is $\mathbf z = \frac1N \sum_{n=1}^N y_n \mathbf x_n \mathbf x_q^\top$. 

If we train $\textsf{MLP} (\mathbf z)$ with the same dataset and initialization, the loss trajectory will be the same as that of $\textsf{ATTN}_{\text M}$.

```bash
python train.py --model mlp --cubic_feat --head 8 --init 1e-6 --show
```

## Attention with Separate Key and Query

Multi-head linear attention with separate key and query, defined as
```math
\textsf{ATTN}_{\text S} (\mathbf X) = \mathbf X + \sum_{i=1}^H \frac1N \mathbf W^V_i \mathbf X \mathbf X^\top {\mathbf W^K_i}^\top \mathbf W^Q_i \mathbf X
```

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
@InProceedings{yedi25icl,
  title = 	 {Training Dynamics of In-Context Learning in Linear Attention},
  author =       {Zhang, Yedi and Singh, Aaditya K. and Latham, Peter E. and Saxe, Andrew},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  year = 	 {2025},
  url = 	 {https://arxiv.org/abs/2501.16265}
}
```
