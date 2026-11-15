import numpy as np
import pandas as pd
import pytest
from PRESS.io.events import normalize_events,map_columns,to_pitch_coordinates
from PRESS import PRESS


def test_explicit_mapping_preserves_unknown_outcomes_and_source():
    raw=pd.DataFrame({'id':[1,2],'kind':['Pass','Mystery'],'frame':[10,20], 'result':[True,None]})
    result=normalize_events(raw,columns={'id':'event_id','kind':'event_type','result':'is_successful'},event_types={'Pass':'pass'})
    assert result.event_type.tolist()==['pass','unknown']
    assert result.original_event.tolist()==['Pass','Mystery']
    assert pd.isna(result.is_successful.iloc[1])
    assert 'event_id' not in raw


def test_invalid_or_ambiguous_inputs_fail():
    with pytest.raises(ValueError):normalize_events(pd.DataFrame({'event_id':[1],'event_type':['pass']}))
    with pytest.raises(ValueError):normalize_events(pd.DataFrame({'event_id':[1],'event_type':['pass'],'frame':[2],'is_successful':['false']}))
    with pytest.raises(ValueError):map_columns(pd.DataFrame({'a':[1],'b':[2]}),{'a':'b'})


def test_coordinate_units_velocity_and_orientation():
    raw=pd.DataFrame({'x':[0,50,100],'y':[0,50,100],'vx':[10,10,10],'vy':[10,10,10]})
    result=to_pitch_coordinates(raw,pitch_length=105,pitch_width=68,x_range=(0,100),y_range=(0,100),y_down=True)
    np.testing.assert_allclose(result.x,[-52.5,0,52.5]);np.testing.assert_allclose(result.y,[34,0,-34])
    np.testing.assert_allclose(result.vx,10.5);np.testing.assert_allclose(result.vy,-6.8)
