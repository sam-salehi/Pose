# --- Train GCNDetector on cached windows ---
import numpy as np
import torch
from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader, Subset

from GCN import (
    BOXINGVI_BONE_PAIRS,
    BOXINGVI_CENTER_JOINT,
    BOXINGVI_GRAPH_EDGES,
    GCNDetector,
    NUM_BOXINGVI_JOINTS,
)
from detector_data import DetectorWindowNpzDataset

_REPO = Path.cwd().resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build one `{ver}_windows.npz` per entry (detector cache cell), then list them here.
DATA_VERS = [f"V{i}" for i in range(1,7)]
DATA_VERS += [f"V{i}" for i in range(8,11)]
_CACHEDIR = _REPO / "Dataset" / "detector_training"
_CACHES = [_CACHEDIR / f"{v}_windows.npz" for v in DATA_VERS]
for p in _CACHES:
    assert p.exists(), f"Missing cache — build it first: {p}"


EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BACKBONE_DROPOUT = 0.1

BATCH_SIZE = 64
VAL_FRAC = 0.15
SEED = 42

torch.manual_seed(SEED)
_parts = [DetectorWindowNpzDataset(p) for p in _CACHES]
video_items = list(zip(DATA_VERS, _parts))
full_ds = ConcatDataset(_parts)
print(f"ConcatDataset: {len(DATA_VERS)} files, {len(full_ds)} windows total")
for v, ds in zip(DATA_VERS, _parts):
    print(f"  {v}: {len(ds)}")

rng = np.random.default_rng(SEED)
n_v = len(video_items)
if n_v < 2:
    raise ValueError(
        "Video-level train/val split requires at least 2 entries in DATA_VERS "
        "(whole workbooks go to train or val — no overlapping adjacent windows). "
        "Add another cache or merge scripts."
    )

# Hold out ~VAL_FRAC of *videos* for validation (at least 1, at most n_v - 1).
n_val_v = max(1, int(round(VAL_FRAC * n_v)))
if n_val_v >= n_v:
    n_val_v = n_v - 1
perm = rng.permutation(n_v)
val_video_idx = set(perm[:n_val_v].tolist())

train_parts = [video_items[i][1] for i in range(n_v) if i not in val_video_idx]
val_parts = [video_items[i][1] for i in range(n_v) if i in val_video_idx]
train_vers = [video_items[i][0] for i in range(n_v) if i not in val_video_idx]
val_vers = [video_items[i][0] for i in range(n_v) if i in val_video_idx]

print(
    f"Video-level split (~{VAL_FRAC:.0%} of workbooks → val; no train/val window leakage):"
)
print(f"  Train videos ({len(train_vers)}): {train_vers}")
print(f"  Val videos ({len(val_vers)}): {val_vers}")

train_concat = ConcatDataset(train_parts)
val_concat = ConcatDataset(val_parts)


def _concat_labels(parts: list) -> np.ndarray:
    return np.concatenate([np.asarray(ds.y, dtype=np.float64) for ds in parts])


def _balanced_window_indices(labels: np.ndarray) -> np.ndarray:
    """Balanced pos/neg subsample; if one class missing, use all windows."""
    pos = np.flatnonzero(labels > 0.5)
    neg = np.flatnonzero(labels <= 0.5)
    n = min(len(pos), len(neg))
    if n == 0:
        return np.arange(len(labels), dtype=np.int64)
    pos = rng.permutation(pos)[:n]
    neg = rng.permutation(neg)[:n]
    out = np.concatenate([pos, neg])
    rng.shuffle(out)
    return out.astype(np.int64)


y_train_pool = _concat_labels(train_parts)
y_val_pool = _concat_labels(val_parts)

train_idx = _balanced_window_indices(y_train_pool)
val_idx = _balanced_window_indices(y_val_pool)

if len(train_idx) == 0:
    raise ValueError("Train split has no windows.")
if len(val_idx) == 0:
    raise ValueError("Val split has no windows.")

n_tr_pos = int(np.sum(y_train_pool[train_idx] > 0.5))
n_val_pos = int(np.sum(y_val_pool[val_idx] > 0.5))

train_ds = Subset(train_concat, train_idx.tolist())
val_ds = Subset(val_concat, val_idx.tolist())

print(
    f"Train (balanced within train videos): punch=yes {n_tr_pos}, punch=no {len(train_idx) - n_tr_pos} "
    f"→ {len(train_ds)} windows"
)
print(
    f"Val (balanced within val videos):   punch=yes {n_val_pos}, punch=no {len(val_idx) - n_val_pos} "
    f"→ {len(val_ds)} windows"
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = GCNDetector(
    in_channels=2,
    num_joints=NUM_BOXINGVI_JOINTS,
    bone_pairs=BOXINGVI_BONE_PAIRS,
    backbone_kwargs={
        "edges": BOXINGVI_GRAPH_EDGES,
        "center": BOXINGVI_CENTER_JOINT,
        "dropout": BACKBONE_DROPOUT,
        "data_bn": True,
    },
    dropout=0,
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = torch.nn.BCELoss()


@torch.no_grad()
def evaluate(loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x)
        total += loss_fn(p, y).item() * y.size(0)
        pred = (p >= 0.5).float()
        correct += (pred == y).sum().item()
        n += y.size(0)
    return total / max(n, 1), correct / max(n, 1)


for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss = 0.0
    run_correct = 0
    run_n = 0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        p = model(x)
        loss = loss_fn(p, y)
        loss.backward()
        opt.step()
        run_loss += loss.item() * y.size(0)
        run_correct += ((p >= 0.5).float() == y).sum().item()
        run_n += y.size(0)

    tr_loss = run_loss / max(run_n, 1)
    tr_acc = run_correct / max(run_n, 1)
    va_loss, va_acc = evaluate(val_dl)
    print(
        f"epoch {epoch:02d}/{EPOCHS}  train loss {tr_loss:.4f} acc {tr_acc:.3f}  "
        f"val loss {va_loss:.4f} acc {va_acc:.3f}"
    )

tr_final_loss, tr_final_acc = evaluate(train_dl)
print(
    f"Train (eval mode, same split): loss {tr_final_loss:.4f} acc {tr_final_acc:.4f}"
)

_det_tag = "_".join(DATA_VERS)
ckpt = _REPO / "checkpoints" / f"gcn_detector_{_det_tag}.pt"
ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state": model.state_dict(),
        "data_vers": DATA_VERS,
        "caches": [str(p) for p in _CACHES],
    },
    ckpt,
)
print(f"Checkpoint → {ckpt.relative_to(_REPO)}")
