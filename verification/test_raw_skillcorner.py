"""Original-file entry point: no precomputed tables or filesystem side effects."""
import hashlib
import numpy as np
import pytest
from PRESS import PRESS
from _fixtures import write_skillcorner_match


def test_loader_derives_velocities_and_leaves_originals_unchanged(tmp_path):
    folder=write_skillcorner_match(tmp_path/'raw')
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    press=PRESS.from_skillcorner(folder)
    assert set(p.name for p in folder.iterdir())==set(before)
    assert all(hashlib.sha256((folder/name).read_bytes()).hexdigest()==value for name,value in before.items())
    np.testing.assert_allclose(press.tracking_data.home_7_vx,.4,atol=1e-12)
    np.testing.assert_allclose(press.tracking_data.away_7_vx,-.2,atol=1e-12)
    np.testing.assert_allclose(press.tracking_data.home_7_vy,0,atol=1e-12)
    assert press.get_frame_context(10)['start_type']=='pass_reception'
    assert press.get_preprocessing_events().event_type.eq('player_possession').all()
    assert not any(press.results.values())


def test_loader_reports_original_files_required(tmp_path):
    folder=tmp_path/'123';folder.mkdir()
    with pytest.raises(FileNotFoundError,match='Original SkillCorner match files missing'):
        PRESS.from_skillcorner(folder)


def test_explicit_match_id_can_differ_from_folder_name(tmp_path):
    folder=write_skillcorner_match(tmp_path/'123')
    moved=tmp_path/'selected_match';folder.rename(moved)
    press=PRESS.from_skillcorner(moved,match_id='123')
    assert press.get_match_info()['match_id']=='123'


def test_loader_uses_input_frequency_for_derivatives(tmp_path):
    folder=write_skillcorner_match(tmp_path/'123')
    press=PRESS.from_skillcorner(folder,fps=20)
    np.testing.assert_allclose(press.tracking_data.home_7_vx,.8,atol=2e-12)
    assert press.fps==20
