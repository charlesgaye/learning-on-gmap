"""Per-dart normalisation (OrbitCNNClassifier(norm=True)).

1. each dart is normalised over ITS channels: mean ~0 (at initialisation: gain 1, bias 0);
2. output equal to a step-by-step reference;
3. gain and bias applied per channel;
4. classifier still invariant to dart renumbering (the norm never mixes darts);
5. norm=False unchanged.

    cd meshcnn-repro
    python -m tests.test_norm
"""
import numpy as np
import torch

from gmapnet.gmap import build_gmap, orbits
from gmapnet.orbitconv import OrbitCNNClassifier, EDGE, FACE, VERTEX
from tests.test_gmap import torus
from tests.test_orbitconv import relabel

ATOL = 1e-4
TYPES = (EDGE, FACE, VERTEX)


def tensors(F):
    g = build_gmap(F)
    alpha = torch.tensor(np.asarray(g.alpha), dtype=torch.long)[None]
    return alpha, [torch.tensor(orbits(g.alpha, t), dtype=torch.long)[None] for t in TYPES]


def reference(model, x, alpha, labs):
    """Step-by-step computation, explicitly normalising each dart over its channels."""
    pre = []
    for conv, ln in zip(model.convs, model.norms):
        h = conv(x, alpha, labs)                                        # (N, C, D)
        mu = h.mean(dim=1, keepdim=True)                                # mean over channels, per dart
        var = h.var(dim=1, unbiased=False, keepdim=True)
        h = (h - mu) / torch.sqrt(var + ln.eps)
        h = h * ln.weight[None, :, None] + ln.bias[None, :, None]       # learned gain / bias, per channel
        pre.append(h)
        x = h.relu()
    return model.head(x.max(dim=2).values), pre


def run():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    alpha, labs = tensors(torus())
    D = alpha.shape[1]
    x = torch.randn(1, 5, D)
    print("Per-dart normalisation:")

    model = OrbitCNNClassifier(5, num_classes=4, widths=(8, 16), dispersion=True, norm=True)
    ref_out, pre = reference(model, x, alpha, labs)
    for k, h in enumerate(pre):
        assert torch.allclose(h.mean(1), torch.zeros_like(h.mean(1)), atol=ATOL), \
            f"layer {k}: after normalisation each dart must have mean ~0 over its channels"
    assert torch.allclose(model(x, alpha, labs), ref_out, atol=ATOL), \
        "output != reference: the norm must act on the CHANNELS of each dart (and keep shape (N, C, D))"
    print("  OK normalisation over the channels of each dart (step-by-step reference)")

    with torch.no_grad():                                               # non-trivial gain / bias
        for ln in model.norms:
            ln.weight.uniform_(0.5, 1.5); ln.bias.uniform_(-0.5, 0.5)
    assert torch.allclose(model(x, alpha, labs), reference(model, x, alpha, labs)[0], atol=ATOL), \
        "with learned gain/bias: gain and bias must apply PER CHANNEL"
    print("  OK gain / bias applied per channel")

    perm = rng.permutation(D)
    x_p = torch.empty_like(x); x_p[:, :, perm] = x                      # dart d becomes perm[d]
    a_p, l_p = relabel(alpha, labs, perm)
    assert torch.allclose(model(x, alpha, labs), model(x_p, a_p, l_p), atol=ATOL), \
        "with norm=True the classifier must stay invariant to dart renumbering"
    print("  OK invariance to renumbering preserved")

    m0 = OrbitCNNClassifier(5, num_classes=4, widths=(8, 16))
    assert m0.norms is None
    m0(x, alpha, labs)
    print("  OK norm=False unchanged")
    print("OK - normalisation verified.")


if __name__ == "__main__":
    run()
