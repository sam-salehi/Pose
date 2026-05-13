#!/usr/bin/env python3
"""
Evaluate a saved binary bio PunchTransformer on one or more workbook versions
(punch clips + gap-sampled no_punch), using the same data contract as
``train_3d_classifier_binary_bio.py`` (TTA mirror averaging, no train-time jitter).

Example::

    python eval_binary_bio_checkpoint.py \\
        --checkpoint checkpoints/punch_transformer_binary_bio_V10_V2_V3_V4_V5_V6_V7_V8_V9.pt \\
        --ver V1
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from punch_transformer import PunchTransformer
from skeleton_constants import H36M_BONE_PAIRS

import train_3d_classifier_binary_bio as binbio


def _parse_versions(s: str) -> frozenset[str]:
    parts = [p.strip().upper() for p in s.replace(",", " ").split() if p.strip()]
    out: list[str] = []
    for p in parts:
        if not p.startswith("V"):
            raise SystemExit(f"Invalid version tag (expected V*): {p!r}")
        out.append(p)
    return frozenset(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval binary bio checkpoint on held-out workbook(s).")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt from train_3d_classifier_binary_bio")
    ap.add_argument(
        "--ver",
        type=str,
        default="V1",
        help='Workbook tag(s), e.g. "V1" or "V1 V2" (default: V1)',
    )
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Not found: {ckpt_path}")

    versions = _parse_versions(args.ver)
    device = binbio.DEVICE

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state")
    if state is None:
        raise SystemExit("Checkpoint missing model_state")

    window = int(ckpt.get("window", binbio.CLF_WINDOW))
    in_ch = int(ckpt.get("in_channels", binbio.IN_CHANNELS))
    classes = list(ckpt.get("punch_classes", binbio.CLASSIFIER_CLASSES))
    model = PunchTransformer(
        num_classes=len(classes),
        in_channels=in_ch,
        edges=H36M_BONE_PAIRS,
        spatial_hidden=int(ckpt.get("spatial_hidden", binbio.MODEL_SPATIAL_HIDDEN)),
        d_model=int(ckpt.get("d_model", binbio.MODEL_D_MODEL)),
        nhead=int(ckpt.get("nhead", binbio.MODEL_NHEAD)),
        num_layers=int(ckpt.get("num_layers", binbio.MODEL_NUM_LAYERS)),
        dim_feedforward=int(ckpt.get("dim_feedforward", binbio.MODEL_DIM_FEEDFORWARD)),
        dropout=float(ckpt.get("dropout", binbio.MODEL_DROPOUT)),
    )
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Unweighted loss — only used inside eval_epoch for the loss scalar; argmax preds unchanged.
    crit = torch.nn.CrossEntropyLoss()

    print(f"Checkpoint: {ckpt_path}")
    print(f"Eval versions: {', '.join(sorted(versions))}  window={window}  device={device}")

    clips, y, used = binbio.load_binary_clips(versions=versions)
    if set(used) != set(versions):
        missing = sorted(set(versions) - set(used))
        if missing:
            raise SystemExit(
                f"No data loaded for {missing}. "
                f"Need Dataset/MotionBERT_3d/{{VER}}/X3D.npy and Dataset/Annotation_files/{{VER}}.xlsx"
            )

    print(f"Clips: {len(y)}  distribution: {dict(Counter(y.tolist()))}")

    ds = binbio.BinaryClipDataset(clips, y, window=window, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    loss, acc, vp, vt, tta_m, tta_r = binbio.eval_epoch(dl, model, crit)
    names = ["Punch", "No Punch"] if len(classes) == 2 else classes

    print(f"\nloss={loss:.4f}  acc={acc:.4f}")
    print(f"TTA argmax mismatch={tta_m:.4f}  mean rel ||Δlogit||={tta_r:.4f}")
    print(binbio._val_summary(vt, vp))
    print("\nClassification report:")
    print(classification_report(vt, vp, target_names=names, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred punch|no_punch):\n", confusion_matrix(vt, vp))


if __name__ == "__main__":
    main()
