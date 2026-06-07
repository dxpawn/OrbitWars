"""Attention-lite distilled student (the fidelity push past the pooling ceiling).

Single explicit-head self-attention over the candidate set per scoring call -> the
most faithful fast approximation of his transformer's cross-candidate attention.
Trains on grouped data, reports held-out per-group top-1, exports a pure-Python
score_many(rows) that runs the SAME forward, with a parity check vs torch.

  python -m rl.distill_train_attn --data rl/distill_grouped/dataset.npz --d 48 --epochs 50
"""
import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _groups(gid):
    order = np.argsort(gid, kind="stable")
    sg = gid[order]
    bounds = np.where(np.diff(sg) != 0)[0] + 1
    return [g for g in np.split(order, bounds) if len(g) >= 1]


def _train_one(X, y, gid, d, epochs, lr, tag, out_dir):
    import torch, torch.nn as nn
    torch.manual_seed(0); np.random.seed(0)
    mean = X.mean(0); std = X.std(0); std[std < 1e-6] = 1.0
    Xn = ((X - mean) / std).astype(np.float32)
    groups = _groups(gid)
    rng = np.random.default_rng(0); rng.shuffle(groups)
    cut = int(0.9 * len(groups)); tr, va = groups[:cut], groups[cut:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class AR(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.emb = nn.Linear(46, d); s.q = nn.Linear(d, d); s.k = nn.Linear(d, d); s.v = nn.Linear(d, d)
            s.h1 = nn.Linear(d, d); s.h2 = nn.Linear(d, 1); s.d = d

        def forward(s, x, mask):  # x (B,K,46) mask (B,K) bool
            h = torch.relu(s.emb(x))
            Q, K, V = s.q(h), s.k(h), s.v(h)
            sc = (Q @ K.transpose(1, 2)) / (s.d ** 0.5)
            sc = sc.masked_fill(~mask[:, None, :], -1e9)
            a = torch.softmax(sc, dim=-1)
            h = h + a @ V
            return s.h2(torch.relu(s.h1(h))).squeeze(-1)  # (B,K)

    model = AR(d).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batch(glist):
        maxk = max(len(g) for g in glist); B = len(glist)
        xb = np.zeros((B, maxk, 46), np.float32); yb = np.zeros((B, maxk), np.float32); mb = np.zeros((B, maxk), bool)
        for i, g in enumerate(glist):
            k = len(g); xb[i, :k] = Xn[g]; yb[i, :k] = y[g]; mb[i, :k] = True
        return (torch.tensor(xb, device=dev), torch.tensor(yb, device=dev), torch.tensor(mb, device=dev))

    BS = 256
    for ep in range(epochs):
        model.train(); rng.shuffle(tr)
        for i in range(0, len(tr), BS):
            gl = tr[i:i + BS]
            xb, yb, mb = batch(gl)
            opt.zero_grad()
            out = model(xb, mb)
            loss = (((out - yb) ** 2) * mb).sum() / mb.sum()
            loss.backward(); opt.step()
    # eval: held-out per-group top1 + R2
    model.eval()
    agree = tot = 0; sse = 0.0; sy = 0.0; ny = 0; ymean_acc = []
    with torch.no_grad():
        for i in range(0, len(va), BS):
            gl = va[i:i + BS]
            xb, yb, mb = batch(gl)
            out = model(xb, mb).cpu().numpy(); ybn = yb.cpu().numpy(); mbn = mb.cpu().numpy()
            for r in range(len(gl)):
                k = mbn[r].sum()
                if k >= 2:
                    p = out[r, :k]; t = ybn[r, :k]
                    agree += int(np.argmax(p) == np.argmax(t)); tot += 1
                sse += float(((out[r, :k] - ybn[r, :k]) ** 2).sum()); ny += int(k)
                ymean_acc.extend(ybn[r, :k].tolist())
    yv = np.array(ymean_acc); r2 = 1.0 - sse / (((yv - yv.mean()) ** 2).sum() + 1e-9)
    t1 = agree / max(1, tot)
    print(f"[{tag}] groups={len(groups)} val_R2={r2:.4f} heldout_top1={t1:.1%} (pooling ~0.85, pointwise ~0.76)", flush=True)
    _export(os.path.join(out_dir, f"student_weights_{tag}.py"), model, mean, std, d, tag, r2, t1, Xn, va, batch, dev)


def _export(path, model, mean, std, d, tag, r2, t1, Xn, va, batch, dev):
    import torch
    P = {n: p.detach().cpu().numpy().astype(np.float64) for n, p in model.named_parameters()}
    def A(a): return repr(a.tolist())
    L = [
        f'"""OUR attention-lite distilled re-ranker ({tag}) - pure-Python score_many drop-in.',
        f"Single-head self-attention over the candidate set. val_R2={r2:.4f} heldout_top1={t1:.3f}.",
        f"Model ours (distilled from his scorer); features reused.", '"""',
        "import math",
        f"D = {d}", f"MEAN = {A(mean)}", f"STD = {A(std.astype(np.float64))}",
        f"EMB_W = {A(P['emb.weight'])}", f"EMB_B = {A(P['emb.bias'])}",
        f"Q_W = {A(P['q.weight'])}", f"Q_B = {A(P['q.bias'])}",
        f"K_W = {A(P['k.weight'])}", f"K_B = {A(P['k.bias'])}",
        f"V_W = {A(P['v.weight'])}", f"V_B = {A(P['v.bias'])}",
        f"H1_W = {A(P['h1.weight'])}", f"H1_B = {A(P['h1.bias'])}",
        f"H2_W = {A(P['h2.weight'])}", f"H2_B = {A(P['h2.bias'])}",
    ]
    L.append(r'''
def _lin(W, B, x):
    out = []
    for r in range(len(W)):
        row = W[r]; acc = B[r]
        for c in range(len(row)): acc += row[c]*x[c]
        out.append(acc)
    return out

def _relu(v): return [x if x>0.0 else 0.0 for x in v]

def score_many(rows):
    k = len(rows)
    if k == 0: return []
    Z = [[(float(r[i])-MEAN[i])/STD[i] for i in range(46)] for r in rows]
    H = [_relu(_lin(EMB_W, EMB_B, z)) for z in Z]
    Q = [_lin(Q_W, Q_B, h) for h in H]
    K = [_lin(K_W, K_B, h) for h in H]
    V = [_lin(V_W, V_B, h) for h in H]
    scale = math.sqrt(D)
    out = []
    for i in range(k):
        sc = [sum(Q[i][t]*K[j][t] for t in range(D))/scale for j in range(k)]
        m = max(sc); ex = [math.exp(s-m) for s in sc]; sm = sum(ex); a = [e/sm for e in ex]
        ctx = [sum(a[j]*V[j][t] for j in range(k)) for t in range(D)]
        h2 = [H[i][t]+ctx[t] for t in range(D)]
        o = _relu(_lin(H1_W, H1_B, h2))
        out.append(_lin(H2_W, H2_B, o)[0])
    return out
''')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    # parity check: pure-Python vs torch on a few val groups
    import importlib.util
    spec = importlib.util.spec_from_file_location("parity_" + tag, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    maxerr = 0.0
    with torch.no_grad():
        for g in va[:20]:
            rows = ((Xn[g] * 0) + Xn[g])  # standardized already? no: score_many re-standardizes from RAW
            # we must feed RAW rows to score_many; reconstruct raw = Xn*std+mean
            raw = (Xn[g] * std + mean)
            py = np.array(mod.score_many(raw.tolist()))
            xb, yb, mb = batch([g])
            tt = model(xb, mb).cpu().numpy()[0, :len(g)]
            maxerr = max(maxerr, float(np.max(np.abs(py - tt))))
    print(f"exported -> {path}  (parity max|py-torch|={maxerr:.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "distill_grouped", "dataset.npz"))
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(_HERE, "student_attn"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dd = np.load(args.data)
    for tag in ("2p", "4p"):
        X, y, gid = dd["X" + tag[0]], dd["y" + tag[0]], dd["g" + tag[0]]
        if len(y) < 100:
            print(f"[{tag}] too few"); continue
        _train_one(X, y, gid, args.d, args.epochs, args.lr, tag, args.out)


if __name__ == "__main__":
    main()
