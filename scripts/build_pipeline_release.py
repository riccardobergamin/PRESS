"""Package the public research source and inventory the distributed objects.

No training, data preprocessing or model inference is performed. Numerical
inputs and fitted research objects stay in the separate local workspace.
"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.release import export_code


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pipeline_archive():
    """Use the pipeline's existing source allowlist, not a recursive ZIP."""
    archive = ROOT / 'pipeline/press-training-pipeline.zip'
    members = []
    with tempfile.TemporaryDirectory(prefix='press-source-release-') as folder:
        source = Path(folder) / 'pipeline'
        export_code(source)
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(source.rglob('*')):
                if not path.is_file():
                    continue
                relative = path.relative_to(source.parent)
                if path.suffix not in {'.py', '.md', '.json'}:
                    raise ValueError(f'Unexpected source archive member: {relative}')
                bundle.write(path, relative)
                members.append(dict(path=str(relative), sha256=checksum(path)))
    record = dict(
        archive=str(archive.relative_to(ROOT)), sha256=checksum(archive),
        file_count=len(members), files=members,
        content='Source, documentation and split/grid protocols only; no numerical inputs or fitted objects.',
    )
    (ROOT / 'docs/pipeline-source-release.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def inventory_objects():
    """Check public data types and record the final models separately."""
    records = []
    allowed_arrays = {'data/examples/expected_threat/xt_grid.npy',
                      'data/examples/bcos/example_field.npz'}
    for folder in ['data', 'PRESS/utils/data']:
        for path in sorted((ROOT / folder).rglob('*')):
            if not path.is_file() or path.name == '.DS_Store':
                continue
            relative = str(path.relative_to(ROOT))
            if 'old' in path.relative_to(ROOT).parts:
                raise ValueError(f'Excluded object in public resources: {relative}')
            if folder == 'data':
                if path.suffix in {'.parquet', '.pkl', '.pickle', '.ubj'}:
                    raise ValueError(f'Individual/fitted research object in public data: {relative}')
                if path.suffix in {'.npy', '.npz'} and relative not in allowed_arrays:
                    raise ValueError(f'Unapproved public array: {relative}')
                if relative in allowed_arrays:
                    role = 'Selected derived spatial example'
                elif relative.startswith('data/benchmarks/'):
                    role = 'Aggregate benchmark/statistical summary or protocol'
                elif path.suffix == '.md':
                    role = 'Resource documentation'
                else:
                    role = 'Descriptive example/report summary or selected-case metadata'
            else:
                role = 'Production inference artifact or fitted feature schema'
            records.append(dict(path=relative, role=role, bytes=path.stat().st_size,
                                sha256=checksum(path)))
    record = dict(
        scope='Final production models and published benchmark/example resources',
        excluded='Raw/processed match tables, full feature matrices, row-level OOF tables, posterior objects, auxiliary ensembles, bootstrap replicates are not included in the release.',
        files=records,
    )
    (ROOT / 'docs/release-objects.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    archive = build_pipeline_archive()
    objects = inventory_objects()
    print(f"Pipeline archive: {archive['file_count']} source/protocol files")
    print(f"Object inventory: {len(objects['files'])} files; no unapproved research objects")


if __name__ == '__main__':
    main()
