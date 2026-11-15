"""Compact storage must change memory use, never fitted predictions."""
import json
import numpy as np
import pytest

from PRESS import PRESS
from PRESS.visualization import plot_pass_map, plot_ball_drive_risk, plot_bcos, plot_risk
from _fixtures import load_test_match


def match():
    tracking, _, actions, _ = load_test_match()
    return PRESS(tracking, actions_data=actions)


def calculate(press, frame=5, **kwargs):
    return press.compute_risk_frame(
        frame, include_residual=True, ppcf_kwargs={'r_max': 12, 'step': 6},
        bcos_kwargs={'r_max': 8, 'step': 4}, **kwargs,
    )


def array_bytes(value):
    if isinstance(value, dict):
        return sum(array_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(array_bytes(item) for item in value)
    return value.nbytes if isinstance(value, np.ndarray) else 0


def test_compaction_preserves_every_scalar_and_discards_fields_after_xloss():
    press = match()
    detailed = calculate(press, keep_details=True)
    detailed_bytes = array_bytes(press.get_results())
    expected_area = press.get_bcos_result(5)['BCOS area']
    compact = calculate(press)
    for name, fields in {
        'BALL_DRIVE_RISK': ['Ball drive risk', 'Survival probability'],
        'ACTION_SELECTION': ['Pass probability', 'Ball drive probability'],
        'HAZARD': ['Hazard', 'band', 'Drive risk', 'Pass failure expected', 'rppcf'],
        'RESIDUAL_RISK': ['Residual loss probability'],
        'XLOSS': ['xLoss', 'Residual loss probability'],
    }.items():
        for field in fields:
            assert compact[name][field] == detailed[name][field]
    assert press.get_bcos_result(5)['BCOS area'] == expected_area
    assert array_bytes(press.get_results()) == 0
    assert detailed_bytes > 1000
    assert 'Features' not in press.get_action_selection_result(5)
    assert compact['ACTION_SELECTION']['context'] == detailed['ACTION_SELECTION']['context']
    assert compact['PPCF']['context'] == detailed['PPCF']['context']
    assert 'params' not in compact['BALL_DRIVE_RISK']
    assert compact['HAZARD']['configuration_id'] == detailed['HAZARD']['configuration_id']
    assert press.compute_xloss(5) == compact['XLOSS']['xLoss']
    assert press.get_hazard_result(5)['band'] == compact['HAZARD']['band']
    assert len(press.result_models) == 5 and len(press.result_configurations) == 1


def test_many_frames_share_configuration_and_models_but_keep_observed_time():
    press = match()
    for frame in [5, 6, 7]:
        calculate(press, frame)
        assert press.get_residual_risk_result(frame)['time_since_start'] == frame / 10
    assert len(press.result_models) == 5 and len(press.result_configurations) == 1
    config = press.get_result_configuration(5)
    config['BCOS']['s'] = 99
    assert press.get_result_configuration(5)['BCOS']['s'] != 99
    press.clear_results(frame_ids=[5])
    assert len(press.result_models) == 5
    press.clear_results()
    assert press.result_models == {} and press.result_configurations == {}


def test_custom_physics_models_and_xt_are_traceable_without_per_frame_copies():
    press = match()
    default = calculate(press)
    first = calculate(press, 6, ball_drive_kwargs={'model': {'s': .35}, 'physics_kwargs': {'v_max': 6.5}},
                      xT_grid=np.full((8, 5), .23))
    second = calculate(press, 7, ball_drive_kwargs={'model': {'s': .4}, 'physics_kwargs': {'v_max': 6.5}})
    first_model = press.get_result_model('BALL_DRIVE_RISK', 6)
    second_model = press.get_result_model('BALL_DRIVE_RISK', 7)
    assert first_model.model_id == second_model.model_id  # names need not be unique
    assert first_model.parameters['s'] == pytest.approx(.35)
    assert second_model.parameters['s'] == pytest.approx(.4)
    assert press.get_result_model('BALL_DRIVE_RISK', 5).parameters['s'] != pytest.approx(.35)
    config = press.get_result_configuration(6)
    assert config['BALL_DRIVE_RISK']['physics_kwargs']['v_max'] == 6.5
    np.testing.assert_array_equal(press.get_xt_grid(6), np.full((8, 5), .23))
    assert 'model_parameters' not in config['BALL_DRIVE_RISK']
    assert len({x['HAZARD']['configuration_id'] for x in [default, first, second]}) == 3
    assert press.get_result_model('PASS_SELECTION', 6) is press.pass_selection_model


def test_plots_require_details_and_never_recompute(monkeypatch):
    import matplotlib.pyplot as plt
    press = match()
    calculate(press)
    def forbidden(*args, **kwargs):
        raise AssertionError('A plot must not trigger inference.')
    monkeypatch.setattr(press, 'compute_risk_frame', forbidden)
    for plot in [plot_pass_map, plot_ball_drive_risk, plot_bcos]:
        with pytest.raises(ValueError, match='keep_details=True'):
            plot(press, 5, show=False)
    fig, _ = plot_risk(press, 5, show=False)
    plt.close(fig)
    with pytest.raises(ValueError, match='keep_details=True'):
        press.get_pass_selection_result(5)
    with pytest.raises(ValueError, match='keep_details=True'):
        press.compute_residual_risk(5)


@pytest.mark.parametrize('detailed', [False, True])
def test_export_restores_models_custom_settings_and_configuration(tmp_path, detailed):
    press = match()
    expected = calculate(press, keep_details=detailed,
                         ball_drive_kwargs={'model': {'s': .35}},
                         xT_grid=np.full((8, 5), .21))
    folder = press.save_results(tmp_path / 'results')
    restored = match()
    restored.load_results(folder)
    assert restored.get_xloss_result(5) == expected['XLOSS']
    assert restored.get_result_model('BALL_DRIVE_RISK', 5).parameters['s'] == pytest.approx(.35)
    np.testing.assert_array_equal(restored.get_xt_grid(5), np.full((8, 5), .21))
    assert restored.compute_xloss(5) == expected['XLOSS']['xLoss']
    assert len(restored.result_models) == 5


def test_export_rejects_dangling_configuration_without_partial_load(tmp_path):
    press = match()
    calculate(press)
    folder = press.save_results(tmp_path / 'results')
    p = folder / 'metadata.json'
    manifest = json.loads(p.read_text())
    manifest['model_settings'] = {}
    p.write_text(json.dumps(manifest))
    restored = match()
    with pytest.raises(ValueError, match='unknown model reference'):
        restored.load_results(folder)
    assert not any(restored.results.values())
    assert not restored.result_models and not restored.result_configurations


def test_version_two_exports_without_shared_registries_still_load(tmp_path):
    press = match()
    press.results['HAZARD'][5] = {'Hazard': .2}
    folder = press.save_results(tmp_path / 'results')
    p = folder / 'metadata.json'
    manifest = json.loads(p.read_text())
    manifest['schema_version'] = 2
    del manifest['configurations'], manifest['model_settings']
    p.write_text(json.dumps(manifest))
    restored = match()
    restored.load_results(folder)
    assert restored.get_hazard_result(5) == {'Hazard': .2}


def test_explicit_bcos_replaces_earlier_map_and_has_its_own_configuration():
    press = match()
    calculate(press, keep_details=True)
    result = calculate(press, bcos_value=12.5)
    assert press.get_bcos_result(5)['BCOS area'] == 12.5
    assert 'grid' not in press.get_bcos_result(5)
    assert press.get_result_configuration(5)['BCOS']['formulation'] == 'explicit'
    assert np.isfinite(result['HAZARD']['Hazard'])


def test_standalone_components_use_shared_configurations_and_fitted_models():
    press = match()
    press.compute_ball_drive_risk(5, model={'s': .35}, physics_kwargs={'v_max': 6.5})
    drive = press.get_ball_drive_risk_result(5)
    assert 'params' not in drive and 'Survival curve' in drive
    settings = press.get_result_configuration(5, 'BALL_DRIVE_RISK')
    assert settings['BALL_DRIVE_RISK']['physics_kwargs']['v_max'] == 6.5
    assert 's' not in settings['BALL_DRIVE_RISK']
    assert press.get_result_model('BALL_DRIVE_RISK', 5).parameters['s'] == pytest.approx(.35)
    # Same settings/model on another frame reuse the same shared objects.
    press.compute_ball_drive_risk(6, model={'s': .35}, physics_kwargs={'v_max': 6.5})
    assert drive['configuration_id'] == press.get_ball_drive_risk_result(6)['configuration_id']
    assert press.get_result_model('BALL_DRIVE_RISK', 5) is press.get_result_model('BALL_DRIVE_RISK', 6)
    press.compute_ppcf(5, r_max=12, step=6, model={'lambda_control': 5., 's': .3})
    settings = press.get_result_configuration(5, 'PPCF')['PPCF']
    assert 'lambda_control' not in settings and 's' not in settings and 'calibration' not in settings
    assert press.get_result_model('PPCF', 5).parameters == {'lambda_control': 5., 's': .3}
    assert 'params' not in press.get_ppcf_result(5)
    press.compute_bcos(5, r_max=4, step=4)
    assert 'params' not in press.get_bcos_result(5)
    assert press.get_result_configuration(5, 'BCOS')['BCOS']['v_bc_max'] == 6.6


def test_standalone_tree_feature_records_are_frame_data_not_model_design():
    press = match()
    press.compute_ppcf(5, r_max=12, step=6)
    press.compute_ball_drive_risk(5)
    press.compute_pass_selection(5)
    press.compute_action_selection_probability(5, bcos_value=1.)
    model = press.get_result_model('ACTION_SELECTION', 5)
    fixed_names = list(model.feature_names)
    record = press.get_action_selection_result(5)['Features']
    assert record['bcos'] == 1.
    assert 'start_type_raw' in record and 'carrier_speed' in record
    press.compute_residual_risk(5)
    press.compute_xloss(5)
    for component in ['PPCF', 'BALL_DRIVE_RISK', 'ACTION_SELECTION', 'RESIDUAL_RISK', 'XLOSS']:
        assert 'params' not in press.get_result(component, 5)
    assert model.feature_names == fixed_names
    assert 'Features' not in press.get_residual_risk_result(5)
    # Changing an upstream component invalidates downstream models/settings too.
    press.compute_ball_drive_risk(5, physics_kwargs={'v_max': 6.5})
    settings = press.get_result_configuration(5)
    assert 'ACTION_SELECTION' not in settings['models']
    assert 'PASS_SELECTION' not in settings['models']
    assert 'RESIDUAL_RISK' not in settings['models']
    with pytest.raises(ValueError, match='compute pass selection'):
        press.get_result_model('PASS_SELECTION', 5)


def test_standalone_configuration_export_roundtrip(tmp_path):
    press = match()
    press.compute_ppcf(5, r_max=12, step=6, model={'lambda_control': 5., 's': .3})
    press.compute_ball_drive_risk(5, model={'s': .35})
    folder = press.save_results(tmp_path / 'standalone')
    restored = match()
    restored.load_results(folder)
    assert restored.get_result_model('PPCF', 5).parameters == {'lambda_control': 5., 's': .3}
    assert restored.get_result_model('BALL_DRIVE_RISK', 5).parameters['s'] == pytest.approx(.35)
    assert restored.get_result_configuration(5) == press.get_result_configuration(5)


def test_complete_risk_computes_only_the_bcos_feature_not_final_transport(monkeypatch):
    from PRESS.spatial_features import BCOS
    press = match()
    # Reproduce the previous execution: explicitly request final transport;
    # action selection then derives its original effective-hazard feature.
    press.compute_ppcf(5, r_max=12, step=6)
    press.compute_ball_drive_risk(5)
    press.compute_pass_selection(5)
    press.compute_bcos(5, r_max=8, step=4)
    assert press.get_result_configuration(5, 'BCOS')['BCOS']['formulation'] == 'transport'
    press.compute_action_selection_probability(5)
    expected_features = press.get_action_selection_result(5)['Features']
    press.compute_residual_risk(5)
    expected_xloss = press.compute_xloss(5)
    expected_h = press.get_hazard_result(5)['Hazard']
    calls = []
    original = BCOS.compute_bcos
    def record(self, *args, **kwargs):
        calls.append(kwargs.get('formulation', 'transport'))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(BCOS, 'compute_bcos', record)
    actual = calculate(press, keep_details=True)
    assert calls == ['effective_hazard']
    assert press.get_result_configuration(5, 'BCOS')['BCOS']['formulation'] == 'effective_hazard'
    assert actual['HAZARD']['Hazard'] == expected_h
    assert actual['XLOSS']['xLoss'] == expected_xloss
    assert actual['ACTION_SELECTION']['Features'] == expected_features
    assert press.get_bcos_result(5)['BCOS area'] == expected_features['bcos']


def test_complete_risk_rejects_a_different_bcos_formulation_before_inference():
    press = match()
    with pytest.raises(ValueError, match='compute_bcos'):
        press.compute_risk_frame(5, bcos_kwargs={'formulation': 'transport'})
    assert not any(press.results.values())
