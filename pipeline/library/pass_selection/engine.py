"""Grouped likelihood engines: observation-sized shards, no test fitting."""
import json
from pathlib import Path
import jax
import numpy as np
import pandas as pd
from pipeline.library.pass_selection.common import sha
from pipeline.library.pass_selection.choice_likelihood import nll, flat_derivatives, reference_derivatives


class Engine:

    def __init__(self,
        membership_root,
        ids,
        sizes,
        chosen,
        fold,
        mode,
        membership_protocol_sha256,
        smoke=False):
        self.blocks = []
        self.mode = mode
        self.count = len(ids)
        self.last_negative_fraction = 0.0
        self.last_seconds = 0.0
        self.keys = ids.copy()
        offset = 0
        row_offset = 0
        for match in ids.match_id.drop_duplicates():
            count = int(ids.match_id.eq(match).sum())
            assert ids.iloc[offset:offset + count].match_id.eq(match).all()
            nc = int(sizes[offset])
            assert (sizes[offset:offset + count] == nc).all()
            if mode == 'cnl':
                folder = Path(membership_root) / f'fold_{fold}' / str(match)
                done = json.loads((folder / 'completion.json').read_text())
                assert done['status'] == 'complete' and (not (folder / 'INCOMPLETE').exists())
                assert smoke or not done['smoke']
                if not smoke:
                    assert done['protocol_sha256'] == membership_protocol_sha256
                path = folder / f'log_alpha_fold_{fold}.npy'
                assert sha(path) == done['files'][path.name]
                keys = pd.read_parquet(folder / 'keys.parquet')
                assert keys.astype(str).reset_index(drop=True).equals(ids.iloc[offset:offset + count][keys.columns].astype(str).reset_index(drop=True))
                alphas = np.load(path, mmap_mode='r')
                assert alphas.shape == (count, nc, 10)
            for start in range(0, count, 32):
                stop = min(start + 32, count)
                a = jax.device_put(np.asarray(alphas[start:stop])) if mode == 'cnl' else None
                y = jax.device_put(np.asarray(chosen[offset + start:offset + stop], np.int64))
                self.blocks.append((slice(offset + start, offset + stop), slice(row_offset + start * nc, row_offset + stop * nc), a, y))
            offset += count
            row_offset += count * nc
        assert offset == self.count
        self.n_rows = row_offset

    def losses(self, scores, rho=1.0):
        output = np.empty(self.count, np.float64)
        for obs, rows, a, y in self.blocks:
            z = np.asarray(scores[rows]).reshape(len(y), -1)
            if self.mode == 'flat':
                values = flat_derivatives(z, y)[0]
            else:
                values = nll(z, a, y, rho)
            output[obs] = np.asarray(values)
        return output

    def gradient(self, scores, rho=1.0):
        import time
        started = time.monotonic()
        g = np.empty(self.n_rows, np.float32)
        h = np.empty(self.n_rows, np.float32)
        negative = 0
        for obs, rows, a, y in self.blocks:
            z = np.asarray(scores[rows]).reshape(len(y), -1)
            if self.mode == 'flat':
                _, gb, hb = flat_derivatives(z, y)
            else:
                _, gb, hb, _ = reference_derivatives(z, a, y, rho)
            g[rows], h[rows] = (np.asarray(gb).ravel(), np.asarray(hb).ravel())
            negative += int((h[rows] < 0).sum())
        assert np.isfinite(g).all() and np.isfinite(h).all()
        self.last_negative_fraction = negative / self.n_rows
        self.last_seconds = time.monotonic() - started
        return (g, h)

    def log_probabilities(self, scores, rho=1.0):
        """Return normalized final log probabilities, preserving candidate order."""
        from scipy.special import logsumexp
        from pipeline.library.pass_selection.choice_likelihood import full_log_probabilities
        output = np.empty(self.n_rows, np.float64)
        for obs, rows, a, y in self.blocks:
            z = np.asarray(scores[rows], np.float64).reshape(len(y), -1)
            values = z - logsumexp(z,
                axis=1,
                keepdims=True) if self.mode == 'flat' else np.asarray(full_log_probabilities(z, a, rho))
            assert np.isfinite(values).all()
            assert np.allclose(logsumexp(values, axis=1), 0.0, atol=2e-10)
            output[rows] = values.ravel()
        return output
