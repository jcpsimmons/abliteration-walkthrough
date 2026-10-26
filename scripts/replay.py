#!/usr/bin/env python3
"""Readable playback of saved stage logs for recording. Never presented as live compute."""
import argparse
from pathlib import Path
import time

from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("step", type=int, choices=range(1, 13))
ap.add_argument("--hold", type=float, default=2.5)
ap.add_argument("--summary", action="store_true", help="Omit batch progress counters from the saved replay")
args = ap.parse_args()
lines = (ROOT / "out" / "logs" / f"{args.step:02}.txt").read_text().splitlines()
if args.summary:
    lines = [line for line in lines if not line.strip().startswith("processed ")]
c = Console(width=110)
for start in range(0, len(lines), 27):
    c.clear()
    c.print(f"SAVED MEASUREMENT / STEP {args.step:02} / PAGE {start//27 + 1}", style="bold yellow")
    c.print(Text.from_ansi("\n".join(lines[start:start+27])), no_wrap=True, overflow="ignore")
    time.sleep(args.hold)
