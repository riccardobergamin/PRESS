"""Explicit field/event mappings for external football data."""
import numpy as np
import pandas as pd

EVENT_TYPES = frozenset({'pass','reception','shot','recovery','interception','tackle',
    'foul_suffered','foul_committed','clearance','restart','stoppage','goal','unknown'})


def map_columns(table,mapping=None,*,required=(),name='table'):
    """Rename provider columns using {source: canonical}; never mutate the input."""
    if not isinstance(table,pd.DataFrame):
        raise TypeError(f'{name} must be a DataFrame.')
    result=table.rename(columns=dict(mapping or {})).copy()
    if result.columns.duplicated().any():
        raise ValueError(f'{name}: mapping creates duplicate column names.')
    missing=set(required)-set(result)
    if missing:
        raise ValueError(f'{name}: missing required columns {sorted(missing)}.')
    return result


def normalize_events(events,*,columns=None,event_types=None):
    """Map events, keeping unknown types and missing results explicit.

    Required: event_id, event_type and either frame, datetime or time_seconds.
    period is required for period-relative seconds. Actor/team and receiver
    are optional for non-action events; segmentation needs an identified actor
    for control/release events. is_successful is nullable, never false-filled.
    A type mapping is explicit {provider_label: canonical_label}; labels not
    in the canonical vocabulary become unknown while original_event is kept.
    """
    result=map_columns(events,columns,required=['event_id','event_type'],name='events')
    if result.event_id.isna().any() or result.event_id.duplicated().any():
        raise ValueError('events.event_id must be nonmissing and unique within the match.')
    if not any(c in result for c in ['frame','datetime','time_seconds']):
        raise ValueError('Events require frame, datetime or period-relative time_seconds.')
    if 'time_seconds' in result and 'datetime' not in result and 'period' not in result:
        raise ValueError('period is required with period-relative event times.')
    if 'original_event' not in result:
        result['original_event']=result.event_type
    if event_types is not None:
        invalid=set(event_types.values())-EVENT_TYPES
        if invalid:
            raise ValueError(f'Unsupported canonical event mapping values: {sorted(invalid)}')
        result['event_type']=result.event_type.map(event_types).fillna('unknown')
    else:result['event_type']=result.event_type.where(result.event_type.isin(EVENT_TYPES),'unknown')
    for c in ['player_id','team_id','receiver_player_id']:
        if c not in result:
            result[c]=None
    if 'is_successful' not in result:
        result['is_successful']=pd.NA
    invalid=result.is_successful.dropna().map(lambda x:not isinstance(x,(bool,np.bool_)))
    if invalid.any():
        raise ValueError('is_successful must contain booleans or missing values; map provider outcomes explicitly.')
    result['is_successful']=result.is_successful.astype('boolean')
    if 'restart_type' not in result:
        result['restart_type']='unknown'
    if 'outcome_verified' not in result:
        result['outcome_verified']=result.is_successful.notna()
    return result


def to_pitch_coordinates(table,*,pitch_length,pitch_width,x_range,y_range,y_down=False):
    """Map declared coordinate bounds to centred pitch metres.

    Required x/y; optional vx/vy are rates in those SAME source units per
    second and receive the same scale/sign transform. Source direction is
    fixed across periods; per-team attacking direction is supplied separately.
    This function does not infer orientation, smooth data or fill positions.
    """
    result=map_columns(table,required=['x','y'],name='coordinates')
    for bounds,extent in [(x_range,pitch_length),(y_range,pitch_width)]:
        if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[1]<=bounds[0] or not np.isfinite(extent) or extent<=0:
            raise ValueError('Coordinate bounds must increase and pitch dimensions must be positive.')
    for axis,bounds,extent,sign in [('x',x_range,pitch_length,1),('y',y_range,pitch_width,-1 if y_down else 1)]:
        scale=sign*extent/(bounds[1]-bounds[0])
        result[axis]=(pd.to_numeric(result[axis],errors='raise')-(bounds[0]+bounds[1])/2)*scale
        if 'v'+axis in result:
            result['v'+axis]=pd.to_numeric(result['v'+axis],errors='raise')*scale
    return result
