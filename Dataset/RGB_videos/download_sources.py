#!/usr/bin/env python3
"""Parse Meta_data.ods for YouTube URLs and download full source videos with yt-dlp."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def cell_link_or_text(cell: ET.Element) -> str:
    """Prefer hyperlink href (avoids duplicate URL when anchor wraps visible text)."""
    for elem in cell.iter():
        if strip_ns(elem.tag) == "a":
            href = elem.get(XLINK_HREF)
            if href:
                return href.replace("&amp;", "&").strip()
    chunks: list[str] = []
    for elem in cell.iter():
        if elem.text:
            chunks.append(elem.text)
        if elem.tail:
            chunks.append(elem.tail)
    return "".join(chunks).replace("&amp;", "&").strip()


def parse_meta_ods(path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))

    rows: list[tuple[str, str]] = []
    for row in root.iter():
        if strip_ns(row.tag) != "table-row":
            continue
        cells = [c for c in row if strip_ns(c.tag) == "table-cell"]
        if len(cells) < 2:
            continue
        label = cell_link_or_text(cells[0])
        link = cell_link_or_text(cells[1])
        if not label or not link:
            continue
        if not re.match(r"^V\d+$", label):
            continue
        if "youtube.com" not in link and "youtu.be" not in link:
            continue
        rows.append((label, link))
    return rows


def main() -> int:
    argv = sys.argv[1:]
    yt_dlp_extra: list[str] = []
    if "--" in argv:
        sep = argv.index("--")
        yt_dlp_extra = argv[sep + 1 :]
        argv = argv[:sep]

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: %(prog)s -o ./source_youtube -- -f bv*+ba/b",
    )
    ap.add_argument(
        "--meta",
        type=Path,
        default=Path(__file__).resolve().parent / "Meta_data.ods",
        help="Path to Meta_data.ods",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "source_youtube",
        help="Directory for downloaded files",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print labels and URLs only; do not call yt-dlp",
    )
    args = ap.parse_args(argv)

    pairs = parse_meta_ods(args.meta)
    if not pairs:
        print("No Video / YouTube rows found in", args.meta, file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label, url in pairs:
        print(f"{label}\t{url}")

    if args.dry_run:
        return 0

    ytdlp = "yt-dlp"
    for label, url in pairs:
        out_tmpl = str(args.output_dir / f"{label}_%(id)s.%(ext)s")
        cmd = [
            ytdlp,
            "-o",
            out_tmpl,
            "--no-overwrites",
            url,
            *yt_dlp_extra,
        ]
        print("\n", " ".join(cmd), "\n", sep="")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
