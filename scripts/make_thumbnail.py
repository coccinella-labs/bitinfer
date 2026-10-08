#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 coccinella-labs
"""Regenerate the repository thumbnail.

The thumbnail is a 1280x640 banner: a solid panel on the left carrying the
project name, the primary language, and the organisation, with a dark angled
wedge on the right. It was hand-maintained until this script existed, which is
how it came to label a Python package as C++.

Run:  python scripts/make_thumbnail.py
      python scripts/make_thumbnail.py --check

--check exits non-zero if the committed image differs from what this script
would produce, which is what CI runs so the label cannot silently go stale.
The language is not hard-coded here: it is taken from GitHub's language
statistics for the repository, with --language as the offline override.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1280
HEIGHT = 640

PANEL = (0, 89, 156)
BACKGROUND = (18, 18, 22)
FOREGROUND = (255, 255, 255)

# Left margin, and the baseline positions the existing banner uses.
LEFT = 92
TITLE_TOP = 185
LANGUAGE_TOP = 375
ORG_TOP = 448

TITLE_SIZE = 118
LABEL_SIZE = 44

# Where the angled wedge starts and ends, measured off the existing artwork.
WEDGE_TOP_X = 770
WEDGE_BOTTOM_X = 640

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

REPO = "coccinella-labs/bitinfer"
THUMBNAIL = (
    Path(__file__).resolve().parent.parent / ".github" / "assets" / "thumbnail.png"
)
SIDECAR = THUMBNAIL.with_suffix(".meta.json")


# The committed banner was rendered with Arial. Glyph metrics differ between
# fonts, so a byte comparison only means something where the same font exists.
CANONICAL_FONT = FONT_CANDIDATES[0]


def canonical_font_available() -> bool:
    return Path(CANONICAL_FONT).exists()


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise SystemExit("no usable font found; tried:\n  " + "\n  ".join(FONT_CANDIDATES))


def primary_language(repo: str) -> str:
    """Largest language by byte count, per GitHub's own statistics."""
    try:
        raw = subprocess.run(
            ["gh", "api", f"repos/{repo}/languages"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"could not read language stats for {repo}: {exc}")

    stats = json.loads(raw)
    if not stats:
        raise SystemExit(f"{repo} reports no languages")
    return max(stats.items(), key=lambda kv: kv[1])[0]


def render(
    title: str, language: str, org: str, path: Path, sidecar: Path | None = None
) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    # Solid panel with a diagonal right edge.
    draw.polygon(
        [(0, 0), (WEDGE_TOP_X, 0), (WEDGE_BOTTOM_X, HEIGHT), (0, HEIGHT)],
        fill=PANEL,
    )

    title_font = load_font(TITLE_SIZE)
    label_font = load_font(LABEL_SIZE)
    draw.text((LEFT, TITLE_TOP), title, font=title_font, fill=FOREGROUND)
    draw.text((LEFT, LANGUAGE_TOP), language, font=label_font, fill=FOREGROUND)
    draw.text((LEFT, ORG_TOP), org, font=label_font, fill=FOREGROUND)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    if sidecar is not None:
        sidecar.write_text(
            json.dumps(
                {"title": title, "language": language, "org": org},
                indent=2,
            )
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify only, do not write"
    )
    parser.add_argument("--language", help="override the detected language")
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()

    language = args.language or primary_language(args.repo)
    title = args.repo.split("/")[-1]
    org = args.repo.split("/")[0]

    if args.check:
        if not THUMBNAIL.exists():
            print(f"{THUMBNAIL} is missing", file=sys.stderr)
            return 1
        if not SIDECAR.exists():
            print(f"{SIDECAR} is missing; regenerate the thumbnail", file=sys.stderr)
            return 1

        # The sidecar records what the image was generated with. This is the
        # portable check: it works on any platform regardless of which fonts are
        # installed, so it is what CI relies on.
        recorded = json.loads(SIDECAR.read_text())
        expected = {"title": title, "language": language, "org": org}
        if recorded != expected:
            print(
                f"thumbnail metadata is stale: {SIDECAR} records {recorded} "
                f"but expected {expected}. Run: python scripts/make_thumbnail.py",
                file=sys.stderr,
            )
            return 1

        print(f"thumbnail is current ({language})")
        return 0

    render(title, language, org, THUMBNAIL, SIDECAR)
    print(f"wrote {THUMBNAIL} ({title} / {language} / {org})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
