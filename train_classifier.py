# --- Train GCNClassifier (6 punch types) from landmarks.npz ---
# Only clips whose (workbook, start, end, label) appear in Dataset/Annotation_files/*.xlsx
# are used — pose in landmarks.npz was extracted for those exact frame ranges at build time.

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from GCN import (
    BOXINGVI_BONE_PAIRS,
    BOXINGVI_CENTER_JOINT,
    BOXINGVI_GRAPH_EDGES,
    GCNClassifier,
    NUM_BOXINGVI_JOINTS,
    PUNCH_CLASSES,
)
from preprocess import _load_annotations, _normalize_label, prepare_windows

_REPO = Path.cwd().resolve()
NPZ_PATH = _REPO / "Dataset" / "landmarks.npz"
_ANNOTATION_DIR = _REPO / "Dataset" / "Annotation_files"


def _all_annotation_workbook_tags(annotation_dir: Path) -> frozenset[str]:
    """Every ``V#`` tag from ``*.xlsx`` stems — train/val use all annotated workbooks."""
    return frozenset(
        p.stem.upper()
        for p in annotation_dir.glob("*.xlsx")
        if p.stem.upper().startswith("V")
    )


def _xlsx_clip_allowlist(annotation_dir: Path) -> set[tuple[str, int, int, str]]:
    """Keys (V#, start_frame, end_frame, normalized_label) from all *.xlsx workbooks."""
    allow: set[tuple[str, int, int, str]] = set()
    for path in sorted(annotation_dir.glob("*.xlsx")):
        stem = path.stem.upper()
        if not stem.startswith("V"):
            continue
        ver = stem
        for s, e, lab in _load_annotations(path):
            allow.add((ver, int(s), int(e), _normalize_label(str(lab))))
    return allow


def _subset_by_versions(
    seqs: np.ndarray,
    labels: np.ndarray,
    versions: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    allowed: frozenset[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep rows whose workbook tag is in ``allowed`` (case-insensitive)."""
    n = len(labels)
    idx = [i for i in range(n) if str(versions[i]).strip().upper() in allowed]
    if not idx:
        raise SystemExit(
            f"No clips left after restricting to versions {sorted(allowed)!r} — "
            "check landmarks.npz ``versions``."
        )
    ix = np.array(idx, dtype=np.int64)
    out_seq = np.empty(len(ix), dtype=object)
    for j, i in enumerate(ix):
        out_seq[j] = seqs[i]
    return (
        out_seq,
        labels[ix],
        versions[ix],
        start_frames[ix],
        end_frames[ix],
    )


def _filter_sequences_by_xlsx(
    seqs: np.ndarray,
    labels: np.ndarray,
    versions: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    allow: set[tuple[str, int, int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only clips that match a row in Annotation_files (same convention as extract_landmarks)."""
    n = len(labels)
    keep: list[int] = []
    for i in range(n):
        key = (
            str(versions[i]),
            int(start_frames[i]),
            int(end_frames[i]),
            _normalize_label(str(labels[i])),
        )
        if key in allow:
            keep.append(i)
    if not keep:
        raise ValueError(
            "No clips left after filtering to Annotation_files — "
            "re-run `python preprocess.py extract` or fix xlsx / landmarks.npz mismatch."
        )
    out_seq = np.empty(len(keep), dtype=object)
    for j, i in enumerate(keep):
        out_seq[j] = seqs[i]
    out_lab = labels[np.array(keep, dtype=np.int64)]
    return out_seq, out_lab

_LABEL_TO_IDX = {
    "Cross": PUNCH_CLASSES.index("cross"),
    "Jab": PUNCH_CLASSES.index("jab"),
    "Lead Hook": PUNCH_CLASSES.index("lead_hook"),
    "Rear Hook": PUNCH_CLASSES.index("rear_hook"),
    "Lead Uppercut": PUNCH_CLASSES.index("lead_uppercut"),
    "Rear Uppercut": PUNCH_CLASSES.index("rear_uppercut"),
}

CLF_WINDOW = 20
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
LR_MIN = 1e-5
WARMUP_EPOCHS = 5
VAL_FRAC = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert NPZ_PATH.exists(), "Missing landmarks.npz — run preprocess extraction first."
if not _ANNOTATION_DIR.is_dir():
    raise SystemExit(f"Missing annotation directory: {_ANNOTATION_DIR}")

TRAIN_VERS = _all_annotation_workbook_tags(_ANNOTATION_DIR)
if not TRAIN_VERS:
    raise SystemExit(f"No V*.xlsx workbooks under {_ANNOTATION_DIR}")

data = np.load(NPZ_PATH, allow_pickle=True)
seqs = data["sequences"]
labels = data["labels"]

for k in ("versions", "start_frames", "end_frames"):
    if k not in data.files:
        raise SystemExit(
            f"landmarks.npz missing {k!r} — rebuild with "
            "`python preprocess.py extract` so clips can be aligned to Annotation_files."
        )

allow = _xlsx_clip_allowlist(_ANNOTATION_DIR)
if not allow:
    raise SystemExit(f"No annotation rows found under {_ANNOTATION_DIR}/*.xlsx")

seqs, labels, versions, starts, ends = _subset_by_versions(
    seqs,
    labels,
    data["versions"],
    data["start_frames"],
    data["end_frames"],
    TRAIN_VERS,
)
print(f"Workbook filter: {sorted(TRAIN_VERS)!r} → {len(labels)} clips from npz.")

seqs, labels = _filter_sequences_by_xlsx(
    seqs,
    labels,
    versions,
    starts,
    ends,
    allow,
)
print(
    f"Annotation alignment: using {len(labels)} clips present in both landmarks.npz "
    f"and {_ANNOTATION_DIR.name}/*.xlsx ({len(allow)} annotated intervals across workbooks)."
)

X_win, y_str = prepare_windows(seqs, labels, window=CLF_WINDOW)
X = np.nan_to_num(X_win.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

keep_rows: list[int] = []
y_list: list[int] = []
for i, s in enumerate(y_str):
    s = str(s).strip()
    if s not in _LABEL_TO_IDX:
        continue
    keep_rows.append(i)
    y_list.append(_LABEL_TO_IDX[s])
X = X[keep_rows]
y_idx = np.array(y_list, dtype=np.int64)
if len(keep_rows) < len(y_str):
    print(f"Kept {len(keep_rows)}/{len(y_str)} clips (dropped unknown labels)")


class ClfDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        t = torch.from_numpy(self.x[i]).unsqueeze(0).unsqueeze(0)
        return t, torch.tensor(self.y[i], dtype=torch.long)


idx_train, idx_val = train_test_split(
    np.arange(len(y_idx)),
    test_size=VAL_FRAC,
    random_state=SEED,
    stratify=y_idx,
)
train_ds = ClfDataset(X[idx_train], y_idx[idx_train])
val_ds = ClfDataset(X[idx_val], y_idx[idx_val])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = GCNClassifier(
    num_classes=len(PUNCH_CLASSES),
    in_channels=2,
    num_joints=NUM_BOXINGVI_JOINTS,
    bone_pairs=BOXINGVI_BONE_PAIRS,
    backbone_kwargs={
        "edges": BOXINGVI_GRAPH_EDGES,
        "center": BOXINGVI_CENTER_JOINT,
        "dropout": 0.1,
        "data_bn": True,
    },
    dropout=0.3,
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
cosine_epochs = max(1, EPOCHS - WARMUP_EPOCHS)
warmup_scheduler = LinearLR(
    opt,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=min(WARMUP_EPOCHS, EPOCHS),
)
cosine_scheduler = CosineAnnealingLR(opt, T_max=cosine_epochs, eta_min=LR_MIN)
sched = SequentialLR(
    opt,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[min(WARMUP_EPOCHS, EPOCHS)],
)
crit = torch.nn.CrossEntropyLoss()


@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    all_p, all_t = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits, yb)
        tot += loss.item() * yb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += yb.size(0)
        all_p.append(pred.cpu())
        all_t.append(yb.cpu())
    all_p = torch.cat(all_p).numpy()
    all_t = torch.cat(all_t).numpy()
    return tot / max(n, 1), correct / max(n, 1), all_p, all_t


best_acc = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, run_ok, run_n = 0.0, 0, 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        run_loss += loss.item() * yb.size(0)
        run_ok += (logits.argmax(1) == yb).sum().item()
        run_n += yb.size(0)
    sched.step()
    lr_now = opt.param_groups[0]["lr"]

    va_loss, va_acc, _, _ = eval_epoch(val_dl)
    if va_acc > best_acc:
        best_acc = va_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"epoch {epoch:02d}/{EPOCHS}  lr {lr_now:.2e}  train loss {run_loss/max(run_n,1):.4f} acc {run_ok/max(run_n,1):.3f}  "
        f"val loss {va_loss:.4f} acc {va_acc:.3f}"
    )

if best_state is not None:
    model.load_state_dict(best_state)
_, _, vp, vt = eval_epoch(val_dl)
print("\nClassification report (val, best checkpoint):")
print(
    classification_report(
        vt,
        vp,
        target_names=[p.replace("_", " ").title() for p in PUNCH_CLASSES],
        zero_division=0,
    )
)
print("Confusion matrix:\n", confusion_matrix(vt, vp))

ckpt = _REPO / "checkpoints" / f"gcn_classifier_{'_'.join(sorted(TRAIN_VERS))}.pt"
ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state": best_state,
        "punch_classes": PUNCH_CLASSES,
        "label_map": _LABEL_TO_IDX,
        "window": CLF_WINDOW,
        "annotation_dir": str(_ANNOTATION_DIR),
        "training_clips": "intersection(landmarks.npz, Annotation_files/*.xlsx)",
        "lr_schedule": {
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup": "LinearLR 0.01→1.0 × base LR",
            "cosine": f"CosineAnnealingLR T_max={max(1, EPOCHS - WARMUP_EPOCHS)} eta_min={LR_MIN}",
        },
    },
    ckpt,
)
print(f"Checkpoint → {ckpt.relative_to(_REPO)}")