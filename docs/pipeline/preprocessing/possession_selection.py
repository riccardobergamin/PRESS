"""Shared SkillCorner boundary/link checks used by the task pipelines."""
from PRESS.io.skillcorner import (
    START_MATCHED, END_MATCHED, is_matched,
    matched_individual_mask, pass_receipt_link_reason, matched_pass_pairs,
    annotate_matched_pass_release_frames,
)


def pass_duration_eligible(release_frame, receipt_frame, *, fps=10):
    """Training flight duration in the inclusive interval [0.5, 10] seconds."""
    gap = receipt_frame - release_frame
    return (gap >= 0.5 * fps) & (gap <= 10 * fps)

