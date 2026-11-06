"""Provider-independent match dimensions and sampling metadata."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)

class MatchInfo:
    """Describe one match in centred metres and seconds.

    ``fps`` converts differences of consecutive frame IDs to seconds. It is
    the sampling frequency of the supplied data, not an integration step.
    ``match_id`` is optional and is never used to infer a data provider.
    """

    fps: float = 10.0
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    match_id: str | None = None

    def __post_init__(self):
        for name in ("fps", "pitch_length", "pitch_width"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.match_id is not None:
            object.__setattr__(self, "match_id", str(self.match_id))

    def to_dict(self):
        """Return a new dictionary suitable for JSON metadata."""
        return asdict(self)
