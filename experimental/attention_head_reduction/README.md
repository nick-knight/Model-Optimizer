# Attention Head Dimensionality Reduction

This research prototype reduces the key and value feature dimensions independently for each
attention head. It follows ESPACE's activation-centric approach: calibrate static principal
subspaces from uncentered activation second moments, emulate the lossy projection while healing a
student model, and absorb the frozen projections into attention weights for deployment.

## Status

The first vertical slice provides:

- exact sample-weighted accumulation of per-head `E[xx^T]` statistics;
- batched eigendecomposition with principal components ordered by decreasing eigenvalue;
- different retained ranks for every head; and
- frozen projection/reconstruction modules for compression-aware training.

Sensitivity-based rank allocation, model adapters, distillation examples, and weight absorption are
planned next. The API is experimental and may change.

## Model Support

| Model/framework | Supported | Notes |
| --- | --- | --- |
| PyTorch attention implementations | Partial | Core operations work with any explicit head axis |
| Bumblebee toy Transformer | Planned | Initial end-to-end integration target |
| Hugging Face Transformers | Planned | Architecture adapters are not implemented |
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

For a basis `P_k` with orthonormal columns, key reconstruction changes the score to
`q^T P_k P_k^T k`. Deployment can therefore replace both query and key coordinates with
`q' = P_k^T q` and `k' = P_k^T k`. Likewise, value reconstruction can be absorbed with
`v' = P_v^T v` and `W_O' = W_O P_v`.

## References

- [ESPACE: Dimensionality Reduction of Activations for Model Compression](https://arxiv.org/abs/2410.05437)
