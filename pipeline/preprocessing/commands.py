"""Raw tracking, matched possessions, canonical bases, then descriptive xT inputs."""
from pathlib import Path
from pipeline.library.common import parser, paths_from
from pipeline.preprocessing import library
from pipeline.library.expected_threat import prepare


def main(stage):
    args = parser('preprocessing', stage, 'Shared base preparation for one fresh run').parse_args()
    if args.base_root is None:
        args.base_root = args.output_root / 'base'
    if stage == 'tracking':
        paths = paths_from(args, require_base=False, require_raw=True)
        matches = args.matches or sorted((p.name for p in paths.raw.iterdir() if p.is_dir()))
        library.tracking(paths.raw, paths.base, matches)
    else:
        paths = paths_from(args)
        matches = args.matches or sorted((p.name for p in (paths.base / 'matches').iterdir() if p.is_dir()))
        if stage == 'context_inputs':
            prepare(paths.base, matches)
        else:
            getattr(library, stage)(paths.base, matches)
