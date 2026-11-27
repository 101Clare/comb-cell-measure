import sys
from pathlib import Path

# Make `hexengine` importable when running pytest from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
