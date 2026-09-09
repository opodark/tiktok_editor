#!/usr/bin/env python3
"""Wrapper per lanciare autoedit senza installare il pacchetto.

Esempi:
    python run.py --media-dir ./foto --audio brano.mp3 --out reel.mp4
    python run.py --media-dir ./foto --audio brano.mp3 --style vhs --cuts-per-beat 2 --seed 7
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoedit.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
