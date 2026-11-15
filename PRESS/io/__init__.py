"""Explicit adapters and safe result persistence; no provider auto-detection."""

from .tracking import convert_long_tracking

__all__ = ["convert_long_tracking"]

from .events import map_columns, normalize_events, to_pitch_coordinates
from .kinematics import differentiate
from .segmentation import canonical_actions, pass_origin_label, segment_possessions
from .synchronization import synchronize_events

__all__ += ["map_columns", "normalize_events", "to_pitch_coordinates",
            "differentiate", "canonical_actions",
            "segment_possessions", "pass_origin_label", "synchronize_events"]
