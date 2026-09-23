"""Release-asset bridge used by the one-command showcase."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from download_release_assets import ensure_release_assets  # noqa: E402,F401
