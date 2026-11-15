"""Period-aware synchronization with explicit matched/unmatched states."""
import numpy as np
import pandas as pd


def synchronize_events(events, tracking, *, fps, already_synchronized=False,
                       max_time_error=0.5):
    """Validate frame links, or match event times to the nearest tracking sample.

    Timestamps are UTC datetimes or period-relative seconds. This temporal
    matcher does not estimate spatial warps or event-specific release timing.
    Use an external synchronizer for that refinement. Unmatched events retain
    a missing frame; receiver, outcome and period are never invented.
    """
    result = events.copy()
    data = tracking.sort_values('frame').copy()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive.')
    if data.frame.duplicated().any():
        raise ValueError('Tracking frame IDs must be unique.')

    if already_synchronized:
        if 'frame' not in result:
            raise ValueError('Already-synchronized events require frame.')
        frames = pd.to_numeric(result.frame, errors='coerce')
        if (result.frame.notna() & frames.isna()).any():
            raise ValueError('Event frame IDs must be numeric or missing.')
        if not (frames.isna() | frames.eq(np.floor(frames))).all():
            raise ValueError('Event frame IDs must be integer or missing.')
        matched = frames.isin(data.frame)
        if 'period' in result and 'period' in data:
            observed_period = frames.map(data.set_index('frame').period)
            matched &= result.period.isna() | observed_period.eq(result.period)
        result['frame'] = frames.astype('Int64')
        result['sync_matched'] = matched
        # Preserve an external synchronizer's more specific provenance.
        if 'sync_method' not in result:
            result['sync_method'] = 'supplied_frame'
        result['sync_reason'] = np.where(
            matched, 'matched', 'missing_frame_or_period_mismatch')
        return result

    if not np.isfinite(max_time_error) or max_time_error < 0:
        raise ValueError('max_time_error must be nonnegative seconds.')
    if 'datetime' in result and 'datetime' in data:
        event_time = pd.to_datetime(result.datetime, utc=True, errors='coerce')
        tracking_time = pd.to_datetime(data.datetime, utc=True, errors='coerce')
        values = event_time.astype('int64').to_numpy() / 1e9
        available = tracking_time.astype('int64').to_numpy() / 1e9
        values[event_time.isna()] = np.nan
        available[tracking_time.isna()] = np.nan
    elif all('time_seconds' in table and 'period' in table for table in [result, data]):
        values = pd.to_numeric(result.time_seconds, errors='coerce').to_numpy(float)
        available = pd.to_numeric(data.time_seconds, errors='coerce').to_numpy(float)
    else:
        raise ValueError('Synchronization requires datetime, or period plus time_seconds, in both tables.')

    matches, errors = [], []
    for index, event in enumerate(result.itertuples()):
        valid = np.isfinite(available)
        if 'period' in result and pd.notna(event.period):
            if 'period' not in data:
                raise ValueError('Tracking period is required to honor event period IDs.')
            valid &= data.period.eq(event.period).to_numpy()
        indices = np.flatnonzero(valid)
        if not len(indices) or not np.isfinite(values[index]):
            matches.append(pd.NA)
            errors.append(np.nan)
            continue
        # Search sorted times inside this period. Ties choose the earlier
        # candidate; no event order correction or extrapolation is inferred.
        order = np.argsort(available[indices], kind='stable')
        indices = indices[order]
        position = np.searchsorted(available[indices], values[index])
        candidates = indices[np.clip([position-1, position], 0, len(indices)-1)]
        best = candidates[np.argmin(abs(available[candidates]-values[index]))]
        error = float(available[best]-values[index])
        errors.append(error)
        matches.append(int(data.iloc[best].frame) if abs(error) <= max_time_error else pd.NA)
    result['frame'] = pd.array(matches, dtype='Int64')
    result['sync_time_error_seconds'] = errors
    result['sync_matched'] = result.frame.notna()
    result['sync_method'] = 'nearest_timestamp'
    result['sync_reason'] = np.where(result.sync_matched, 'matched', 'no_frame_within_tolerance')
    return result
