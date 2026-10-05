"""Dead-ReLU diagnostic: fraction of active units per layer during the first training steps.

    cd meshcnn-repro
    python -m gmapnet.diag_relu E,F,V+d        # without normalisation: layer 2 dies
    python -m gmapnet.diag_relu E,F,V+d+n      # with the per-dart LayerNorm
"""
import sys

import numpy as np
import torch

from gmapnet.data import list_classes, load_split, with_labels
from gmapnet.orbitconv import OrbitCNNClassifier
from gmapnet.train_orbit import parse_code
from src.train import standardize

STEPS = 150


def active_fractions(model, x, alpha, labs):
    """Fraction of values > 0 at the output of each ReLU."""
    _, acts = model(x[None], alpha[None], [l[None] for l in labs], return_acts=True)
    return [round((a > 0).float().mean().item(), 3) for a in acts]


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "E,F,V+d"
    types, dispersion, norm = parse_code(code)
    classes = list_classes()
    tr = load_split("train", classes)
    allx = torch.cat([x for x, _, _ in tr], 1)
    items = with_labels(standardize(tr, allx.mean(1, keepdim=True), allx.std(1, keepdim=True) + 1e-6), types)
    torch.manual_seed(0); np.random.seed(0)
    model = OrbitCNNClassifier(3, len(classes), (64, 128, 256), orbit_types=types,
                               dispersion=dispersion, norm=norm)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for step, i in enumerate(np.random.permutation(len(items))[:STEPS]):
        x, alpha, labs, y = items[i]
        if step in (0, 20, 60, STEPS - 1):
            with torch.no_grad():
                print(f"{code} step {step:3d}: active units per layer {active_fractions(model, x, alpha, labs)}")
        loss = torch.nn.functional.cross_entropy(model(x[None], alpha[None], [l[None] for l in labs]),
                                                 torch.tensor([y]))
        opt.zero_grad(); loss.backward(); opt.step()


if __name__ == "__main__":
    main()
