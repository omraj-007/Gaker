from __future__ import annotations

import sys
from pathlib import Path


def add_project_root_to_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root
