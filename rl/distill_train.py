"""Train OUR re-ranker (the deliverable) to match the friend's scores via distillation.

Input: rl/distill_data/dataset.npz  (X2,y2,X4,y4) = (46 features -> his raw logit).
For each mode (2p/4p): standardize, train a small MLP (46->H->H->1, ReLU, MSE on his
logit), report val R²/Pearson (high R² => same candidate ranking within a state),
save a torch checkpoint AND export a pure-Python `student_weights_<mode>.py` whose
`score_many(rows)` is a math-only drop-in for his feature46_weights_<mode>.score_many.

  python -m rl.distill_train --data rl/distill_data/dataset.npz --hidden 64 --epochs 40
"""
import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _train_one(X, y, hidden, epochs, lr, batch, seed, tag):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(y)
    mean = X.mean(0); std = X.std(0); std[std < 1e-6] = 1.0
    Xn = (X - mean) / std
    idx = np.random.permutation(n)
    cut = int(0.9 * n)
    tr, va = idx[:cut], idx[cut:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    model = nn.Sequential(
        nn.Linear(46, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xtr, ytr = Xt[tr].to(dev), yt[tr].to(dev)
    Xva, yva = Xt[va].to(dev), yt[va].to(dev)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xtr[b])
            loss = lossf(out, ytr[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pv = model(Xva).cpu().numpy().ravel()
    yv = yva.cpu().numpy().ravel()
    ss_res = float(((yv - pv) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-9)
    pear = float(np.corrcoef(pv, yv)[0, 1])
    print(f"[{tag}] n={n} val_R2={r2:.4f} pearson={pear:.4f} val_mse={ss_res/len(yv):.4f}", flush=True)
    # extract numpy weights
    W = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            W.append((layer.weight.detach().cpu().numpy().astype(np.float64),
                      layer.bias.detach().cpu().numpy().astype(np.float64)))
    return W, mean.astype(np.float64), std.astype(np.float64), r2, pear


def _export_py(path, W, mean, std, tag, r2, pear):
    def arr(a):
        return repr(a.tolist())
    lines = [
        f'"""OUR distilled re-ranker ({tag}) - pure-Python drop-in for score_many.',
        f"Trained by knowledge distillation from the friend's scorer (features reused, model ours).",
        f"val R2={r2:.4f} pearson={pear:.4f}. MLP 46->{W[0][0].shape[0]}->{W[1][0].shape[0]}->1 ReLU.",
        '"""',
        "import math",
        f"MEAN = {arr(mean)}",
        f"STD = {arr(std)}",
    ]
    for i, (w, b) in enumerate(W):
        lines.append(f"W{i} = {arr(w)}")
        lines.append(f"B{i} = {arr(b)}")
    lines.append(f"_NL = {len(W)}")
    lines.append("""
def _fwd(x):
    # standardize
    z = [(x[i] - MEAN[i]) / STD[i] for i in range(len(x))]
    for li in range(_NL):
        W = globals()['W%d' % li]; B = globals()['B%d' % li]
        out = []
        for r in range(len(W)):
            row = W[r]
            acc = B[r]
            for c in range(len(row)):
                acc += row[c] * z[c]
            out.append(acc)
        if li < _NL - 1:
            out = [v if v > 0.0 else 0.0 for v in out]  # ReLU
        z = out
    return z[0]


def score_many(rows):
    return [_fwd([float(v) for v in r]) for r in rows]
""")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"exported -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "distill_data", "dataset.npz"))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(_HERE, "student"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.data)
    for tag, Xk, yk in [("2p", "X2", "y2"), ("4p", "X4", "y4")]:
        X, y = d[Xk], d[yk]
        if len(y) < 100:
            print(f"[{tag}] too few rows ({len(y)}), skipping"); continue
        W, mean, std, r2, pear = _train_one(
            X, y, args.hidden, args.epochs, args.lr, args.batch, seed=0, tag=tag)
        _export_py(os.path.join(args.out, f"student_weights_{tag}.py"), W, mean, std, tag, r2, pear)


if __name__ == "__main__":
    main()
