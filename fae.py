"""
Frame Angular Encoding (FAE) — Steps 1–9 from math.md.

Maps windowed pose clips (N, 20, 12, 2) in pixel space to (N, 192) feature vectors.
Implementation lives in ``paper_features.encode_fae_*``.
"""

from __future__ import annotations

from paper_features import encode_fae_flat, encode_fae_timesteps

encode_fae = encode_fae_flat

__all__ = ["encode_fae", "encode_fae_timesteps", "encode_fae_flat"]
