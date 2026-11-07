"""Score held-out composition and bootstrap the fixed risk bands by match."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.library.common import parser, paths_from
from pipeline.composed_risk.library import score
if __name__ == '__main__':
    args = parser('composed_risk', 'scores', __doc__).parse_args()
    score(paths_from(args))
