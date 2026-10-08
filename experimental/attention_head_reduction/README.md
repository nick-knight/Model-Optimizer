# Attention Head Dimensionality Reduction

This research prototype reduces the key and value feature dimensions independently for each
attention head. It supports activation-MSE PCA and gradient-weighted squared-loss-gap and output-KL
objectives, emulates the lossy projection while healing a student model, and is intended to absorb
the frozen low-rank maps into attention weights for deployment.

## Status

The first vertical slice provides:

- exact sample-weighted accumulation of per-head `E[xx^T]` statistics;
- paired accumulation of per-head activation and loss-gradient second moments;
- batched eigendecomposition with principal components ordered by decreasing eigenvalue;
- different retained ranks for every head; and
- frozen orthogonal or oblique projection/reconstruction modules for compression-aware training;
- adapters for flattened Hugging Face-style key/value linear projections.

Sensitivity-based rank allocation, architecture-specific orchestration, and weight absorption are
planned next. The API is experimental and may change.

## Model Support

| Model/framework | Supported | Notes |
| --- | --- | --- |
| PyTorch attention implementations | Partial | Core operations work with any explicit head axis |
| Bumblebee toy Transformer | Yes | End-to-end research integration in Pulsar |
| Hugging Face Transformers | Partial | Flattened K/V linear-output adapter; tested with Nemotron-H |
| Megatron-Core | Planned | Architecture adapters are not implemented |

## Deployment

| Framework | Supported | Notes |
| --- | --- | --- |
| PyTorch eager | Emulation only | Projection and reconstruction retain original tensor shapes |
| TensorRT-LLM | No | Variable per-head dimensions require an export representation and kernels |
| vLLM | No | Variable per-head dimensions require an export representation and kernels |
| SGLang | No | Variable per-head dimensions require an export representation and kernels |

## Usage

```python
from experimental.attention_head_reduction import HeadProjector, SecondMomentAccumulator

stats = SecondMomentAccumulator(num_heads=8, head_dim=64)
for keys in calibration_keys:  # for example: (batch, heads, sequence, head_dim)
    stats.update(keys, head_axis=1)

projector = HeadProjector.from_second_moment(stats.compute(), ranks=[32] * 8)
approximated_keys = projector(keys, head_axis=1)
```

`SecondMomentAccumulator` deliberately does not subtract the activation mean. This matches ESPACE's
minimum-MSE construction, which uses the eigenspace of the uncentered autocorrelation matrix.
Projection matrices are PyTorch buffers rather than parameters, so gradients flow through the
approximation while the calibrated subspaces remain fixed.

For the squared-loss-gap objective, calibration additionally accumulates `G = E[gg^T]`, where
`g` is the loss gradient with respect to the activation. Under a K-FAC factorization, the local
first-order objective for reconstruction `M` is

```text
tr(G (I - M) E[xx^T] (I - M)^T).
```

The implementation performs PCA on `G^(1/2) E[xx^T] G^(1/2)`. Its compression and reconstruction
bases are distinct and implement a projection that is orthogonal in the damped gradient metric,
but generally oblique in Euclidean coordinates:

```python
from experimental.attention_head_reduction import (
    ActivationGradientCalibration,
    HeadProjector,
)

calibration = ActivationGradientCalibration(num_heads=8, head_dim=64)
outputs = model_with_calibration(inputs)
loss = criterion(outputs, targets)
loss.backward()
activation_moment, gradient_moment = calibration.compute()
projector = HeadProjector.from_squared_loss_gap(
    activation_moment,
    gradient_moment,
    ranks=[32] * 8,
)
```

For packed next-token calibration, callers can use `set_activation_weights()` to count an
activation once for every target prefix containing its position. Exact target-specific backwards
reuse the forward graph and call `set_gradient_normalizer()` before each backward. Random-sign
vector-Jacobian products provide an unbiased alternative that also reuses the graph. Flattening
token positions while accumulating the two moments drops cross-context-position terms; using a
separate calibrator at each K/V site drops cross-site terms.

For `E[D_KL(p || p_hat)]`, the local second-order term uses the model Fisher. At unit temperature,
it can be accumulated from score gradients of pseudo-labels drawn independently from `p`. Exact
calibration sums the score outer products over the vocabulary; sampled calibration uses one packed
backward per probe. Combining independently sampled labels for multiple next-token targets assumes
their cross-target score-gradient products cancel in expectation. A finite number of probes does
not cancel them exactly, so this approximation should be checked empirically if it becomes a
material source of error. Both KL and squared-loss-gap moments use the same gradient-weighted
eigensolver and generally produce oblique projections.

Gradient-weighted solves use float64 CPU linear algebra after independently normalizing the two
moments by their mean eigenvalue. Each nominally PSD matrix is symmetrized and checked for finite
values and material negative eigenvalues; only roundoff-scale negative eigenvalues are clamped.
The gradient metric is relatively damped and capped at a configurable condition number. A
Cholesky factor and triangular solve avoid explicitly forming its inverse square root. The solver
also checks full-rank biorthogonality and projection reconstruction before returning the bases.
`gradient_weighted_components_with_diagnostics()` reports the original scales, eigenvalue clamps,
condition-number floor, zero-metric fallback, Cholesky jitter, and invariant residuals.

On Apple MPS, batch Gram matrices are computed in float32 on-device, then the much smaller head
matrices are transferred to CPU and accumulated in float64. Other devices retain float64
accumulation by default.

For compression basis `C_k` and reconstruction basis `R_k`, key reconstruction changes the score
to `q^T R_k C_k^T k`. Deployment can replace the coordinates with `q' = R_k^T q` and
`k' = C_k^T k`. Likewise, value reconstruction uses `v' = C_v^T v` and absorbs `R_v` into
`W_O`. Orthogonal PCA is the special case `C = R`.

## References

- [ESPACE: Dimensionality Reduction of Activations for Model Compression](https://arxiv.org/abs/2410.05437)
