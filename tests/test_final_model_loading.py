"""Production readers fail explicitly; navigation does not load choice weights."""
import pytest

from PRESS import PRESS
from PRESS.model import ActionSelectionModel, PassSelectionModel
from _fixtures import load_test_match


@pytest.mark.parametrize('reader, resolver', [
    (ActionSelectionModel, '_action_model'),
    (PassSelectionModel, '_choice_model'),
])
def test_missing_final_weights_raise_without_a_historical_fallback(monkeypatch, reader, resolver):
    tracking, *_ = load_test_match()
    press = PRESS(tracking)

    def missing(cls):
        raise FileNotFoundError('Final model unavailable')

    monkeypatch.setattr(reader, 'pretrained', classmethod(missing))
    with pytest.raises(FileNotFoundError, match='Final model unavailable'):
        getattr(press, resolver)()


def test_navigation_is_independent_of_model_loading(monkeypatch):
    tracking, possessions, actions, _ = load_test_match()
    press = PRESS(tracking, possessions_data=possessions, actions_data=actions)

    def unavailable(cls):
        raise AssertionError('Navigation must not load choice models')

    monkeypatch.setattr(ActionSelectionModel, 'pretrained', classmethod(unavailable))
    monkeypatch.setattr(PassSelectionModel, 'pretrained', classmethod(unavailable))
    assert not press.get_checkpoints().empty
    assert press.get_frame_context(5)['start_type'] == 'pass_reception'
    assert press.action_selection_model is None
    assert press.pass_selection_model is None
