"""Context-aware distilled student (Edge A).

Each candidate's 46 features are augmented with its SET's mean+max (46+46+46=138) so a
still-pointwise-at-inference MLP can see how a candidate compares to the others in its
call -- approximating his transformer's cross-candidate attention. Trains on grouped
data, reports per-GROUP top-1 agreement (the re-rank metric) on held-out groups, and
exports a pure-Python score_many(rows) that recomputes the set context from `rows`.

  python -m rl.distill_train_ctx --data rl/distill_grouped/dataset.npz --hidden 96 --epochs 60
"""
import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _build_ctx(X, gid):
    uniq, inv = np.unique(gid, return_inverse=True)
    G = len(uniq)
    sums = np.zeros((G, 46), np.float64)
    sq = np.zeros((G, 46), np.float64)
    cnt = np.zeros(G, np.float64)
    np.add.at(sums, inv, X)
    np.add.at(sq, inv, X.astype(np.float64) ** 2)
    np.add.at(cnt, inv, 1.0)
    means = (sums / cnt[:, None]).astype(np.float32)
    var = (sq / cnt[:, None]) - (sums / cnt[:, None]) ** 2
    stds = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    maxs = np.full((G, 46), -1e30, np.float32)
    np.maximum.at(maxs, inv, X)
    mins = np.full((G, 46), 1e30, np.float32)
    np.minimum.at(mins, inv, X)
    # per-candidate: own 46 + set mean/max/min/std (46 each) = 230
    Xc = np.concatenate([X, means[inv], maxs[inv], mins[inv], stds[inv]], axis=1)
    return Xc, inv, G


def _group_top1(pred, y, inv, G):
    # per group, does argmax(pred) match argmax(y)? only groups with >=2 candidates
    best_p = np.full(G, -1e30); arg_p = np.full(G, -1, np.int64)
    best_y = np.full(G, -1e30); arg_y = np.full(G, -1, np.int64)
    cnt = np.zeros(G, np.int64)
    for i in range(len(pred)):
        g = inv[i]; cnt[g] += 1
        if pred[i] > best_p[g]: best_p[g] = pred[i]; arg_p[g] = i
        if y[i] > best_y[g]: best_y[g] = y[i]; arg_y[g] = i
    mask = cnt >= 2
    return float(np.mean(arg_p[mask] == arg_y[mask]))


def _train_one(X, y, gid, hidden, epochs, lr, batch, tag, out_dir):
    import torch, torch.nn as nn
    torch.manual_seed(0); np.random.seed(0)
    Xc, inv, G = _build_ctx(X, gid)
    mean = Xc.mean(0); std = Xc.std(0); std[std < 1e-6] = 1.0
    Xn = (Xc - mean) / std
    # split by GROUP
    perm = np.random.permutation(G); cut = int(0.9 * G)
    tr_groups = set(perm[:cut].tolist())
    is_tr = np.array([g in tr_groups for g in inv])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xn, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    D = Xc.shape[1]
    model = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.MSELoss()
    tr_idx = np.where(is_tr)[0]; va_idx = np.where(~is_tr)[0]
    Xtr = Xt[tr_idx].to(dev); ytr = yt[tr_idx].to(dev)
    for ep in range(epochs):
        model.train()
        p = torch.randperm(len(tr_idx), device=dev)
        for i in range(0, len(tr_idx), batch):
            b = p[i:i + batch]; opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pv = model(Xt.to(dev)).cpu().numpy().ravel()
    # held-out top1 (val groups only)
    t1 = _group_top1_split(pv, y, inv, va_idx)
    # R2 on val rows
    yv = y[va_idx]; pvv = pv[va_idx]
    r2 = 1.0 - float(((yv - pvv) ** 2).sum()) / (float(((yv - yv.mean()) ** 2).sum()) + 1e-9)
    print(f"[{tag}] groups={G} val_R2={r2:.4f} heldout_top1={t1:.1%} (pointwise baseline ~0.76)", flush=True)
    W = [(l.weight.detach().cpu().numpy().astype(np.float64), l.bias.detach().cpu().numpy().astype(np.float64))
         for l in model if isinstance(l, nn.Linear)]
    _export(os.path.join(out_dir, f"student_weights_{tag}.py"), W, mean.astype(np.float64), std.astype(np.float64), tag, r2, t1)


def _group_top1_split(pred, y, inv, va_idx):
    # restrict to val rows, regroup
    sub_gid = inv[va_idx]
    uniq, inv2 = np.unique(sub_gid, return_inverse=True)
    return _group_top1(pred[va_idx], y[va_idx], inv2, len(uniq))


def _reindex(inv, mask):
    return inv[mask]


def _export(path, W, mean, std, tag, r2, t1):
    def arr(a): return repr(a.tolist())
    L = [
        f'"""OUR context-aware distilled re-ranker ({tag}) - pure-Python score_many drop-in.',
        f"Per-candidate 46 feats + set mean(46) + set max(46) = 138 -> MLP. Distilled from his scorer.",
        f"val_R2={r2:.4f} heldout_top1={t1:.3f}. Model ours; features reused.",
        '"""',
        "import math",
        f"MEAN = {arr(mean)}", f"STD = {arr(std)}",
    ]
    for i, (w, b) in enumerate(W):
        L.append(f"W{i} = {arr(w)}"); L.append(f"B{i} = {arr(b)}")
    L.append(f"_NL = {len(W)}")
    L.append('''
def score_many(rows):
    rows = [[float(v) for v in r] for r in rows]
    k = len(rows)
    if k == 0:
        return []
    n = 46
    mean = [0.0]*n; mx = [-1e30]*n; mn = [1e30]*n; sq = [0.0]*n
    for r in rows:
        for j in range(n):
            v = r[j]
            mean[j] += v; sq[j] += v*v
            if v > mx[j]: mx[j] = v
            if v < mn[j]: mn[j] = v
    sd = [0.0]*n
    for j in range(n):
        mean[j] /= k
        var = sq[j]/k - mean[j]*mean[j]
        sd[j] = var**0.5 if var > 0.0 else 0.0
    out = []
    for r in rows:
        x = r + mean + mx + mn + sd  # 230
        z = [(x[i]-MEAN[i])/STD[i] for i in range(len(x))]
        for li in range(_NL):
            W = globals()['W%d'%li]; B = globals()['B%d'%li]
            o = []
            for rr in range(len(W)):
                row = W[rr]; acc = B[rr]
                for c in range(len(row)): acc += row[c]*z[c]
                o.append(acc)
            if li < _NL-1:
                o = [v if v>0.0 else 0.0 for v in o]
            z = o
        out.append(z[0])
    return out
''')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"exported -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "distill_grouped", "dataset.npz"))
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(_HERE, "student_ctx"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.data)
    for tag in ("2p", "4p"):
        X, y, gid = d["X" + tag[0]], d["y" + tag[0]], d["g" + tag[0]]
        if len(y) < 100:
            print(f"[{tag}] too few"); continue
        _train_one(X, y, gid, args.hidden, args.epochs, args.lr, args.batch, tag, args.out)


if __name__ == "__main__":
    main()
