"""Pass selection: training stage of the final pipeline."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.library.cli import task_main
if __name__ == '__main__':
    task_main('pass_selection', 'training')
