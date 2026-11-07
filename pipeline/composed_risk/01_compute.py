"""Compose held-out component predictions on verified canonical states."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.library.common import parser, paths_from
from pipeline.composed_risk.library import compute
if __name__ == '__main__':
    args = parser('composed_risk', 'features', __doc__).parse_args()
    compute(paths_from(args), args.threads)
