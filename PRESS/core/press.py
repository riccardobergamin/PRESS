"""One-match interface to PRESS, independent of the tracking provider."""

from pathlib import Path

import numpy as np
import pandas as pd

from .computation import ComputationMixin
from .metadata import MatchInfo
from .navigation import NavigationMixin
from .results import ResultsMixin


class PRESS(NavigationMixin, ComputationMixin, ResultsMixin):
    """Load one match, navigate its states and evaluate possession risk.

    Use canonical tracking tables directly, or ``from_skillcorner`` for
    original SkillCorner match files. Initialization validates/copies inputs; it runs no predictions.
    Individual/team possessions and action annotations are optional for
    explicit-context frame calculations. Model instances or exported manifest
    paths can replace the bundled pretrained models.
    """

    def __init__(
        self,
        tracking_data,
        events_data=None,
        *,
        possessions_data=None,
        actions_data=None,
        player_possessions_data=None,
        ball_drive_model=None,
        pass_risk_model=None,
        pass_selection_model=None,
        action_selection_model=None,
        residual_risk_model=None,
        fps=10.0,
        pitch_length=None,
        pitch_width=None,
        match_id=None,
        player_metadata=None,
        xT_grid=None,
    ):
        """Initialize canonical centred-metre tracking for one match.

        Parameters
        ----------
        tracking_data : pandas.DataFrame
            Unique integer ``frame``; ``player_possession`` (home/away alias
            or missing), ``attacking_side``, ``ball_x, ball_y`` and player
            columns ``<alias>_x, _y, _vx, _vy``. Speed ``_v`` is optional and
            is derived from the supplied velocity components when absent.
        events_data : pandas.DataFrame, optional
            Canonical events, not a provider's raw event table.
        possessions_data : pandas.DataFrame, optional
            Explicit team intervals with ``possession_id, team_id,
            frame_start, frame_end``. Endpoints are inclusive.
        player_possessions_data : pandas.DataFrame, optional
            Individual intervals and verified pass links. Alternative to
            team intervals; see ``link_team_possessions`` in the reference.
        actions_data : pandas.DataFrame, optional
            Canonical action intervals with ``frame_start, frame_end,
            action, start_type, previous_action``. When omitted and individual
            possessions are supplied, PRESS constructs actions using its
            canonical one-second carry/pass convention.
        fps : float
            Input frame frequency. Default 10 Hz preserves the original
            dataset; use the actual frequency for another tracking source.
        pitch_length, pitch_width : float, optional
            Native pitch dimensions in metres. Otherwise use constant input
            columns, or 105 by 68 m when dimensions are not supplied.
        match_id : str or int, optional
            User/provider match identifier, retained in metadata and exports.
        player_metadata : pandas.DataFrame, optional
            Roster keyed by canonical ``player_id``; source IDs/names may be
            included. The long-table conversion helper returns the source-ID mapping.
        xT_grid : array-like, optional
            Value surface (n_x, n_y), oriented to attack toward positive x.
            Used by all contextual models. None selects the bundled surface.
        ball_drive_model, pass_risk_model, pass_selection_model,
        action_selection_model, residual_risk_model : model or path, optional
            Explicit fitted model or JSON-manifest path. The two physical
            models also accept named parameter dictionaries. Bundled production
            models are resolved lazily when their component is requested.
        """
        from PRESS.io.validation import prepare_tracking
        from PRESS.utils.match import link_team_possessions, validate_intervals
        from PRESS.utils.sanitizers import validate_events_data
        from PRESS.model import (
            ActionSelectionModel,
            BallDriveModel,
            PassRiskModel,
            PassSelectionModel,
            ResidualRiskModel,
        )

        self.tracking_data, self.match_info = prepare_tracking(
            tracking_data, fps=fps, pitch_length=pitch_length,
            pitch_width=pitch_width, match_id=match_id,
        )
        if events_data is not None:
            validate_events_data(events_data)
        self.events_data = None if events_data is None else events_data.copy()
        self.ball_drive_model = self._load_model(ball_drive_model, BallDriveModel)
        self.pass_risk_model = self._load_model(pass_risk_model, PassRiskModel)
        self.pass_selection_model = self._load_model(pass_selection_model, PassSelectionModel)
        self.action_selection_model = self._load_model(action_selection_model, ActionSelectionModel)
        self.residual_risk_model = self._load_model(residual_risk_model, ResidualRiskModel)

        from PRESS.utils.state import validate_xt_grid
        self.xT_grid = None if xT_grid is None else validate_xt_grid(xT_grid)
        self.player_metadata = self._prepare_roster(player_metadata)
        self.player_possessions_data = None
        if player_possessions_data is not None:
            if possessions_data is not None:
                raise ValueError("Supply team intervals or player intervals, not both.")
            possessions_data, self.player_possessions_data = link_team_possessions(
                player_possessions_data
            )
        self.possessions_data = None
        if possessions_data is not None:
            table = validate_intervals(possessions_data, [
                "possession_id", "team_id", "frame_start", "frame_end",
            ])
            if table.possession_id.isna().any() or table.possession_id.duplicated().any():
                raise ValueError("possession_id must be nonmissing and unique within a match.")
            if not table.team_id.isin(["home", "away"]).all():
                raise ValueError("team_id must be home or away.")
            if (
                len(table) > 1
                and np.any(table.frame_start.to_numpy()[1:] <= table.frame_end.to_numpy()[:-1])
            ):
                raise ValueError("Team possession intervals must not overlap.")
            self.possessions_data = table
        self.actions_data = None
        if actions_data is None and self.player_possessions_data is not None:
            from PRESS.io.segmentation import canonical_actions

            # Possession boundaries belong to the input dataset. The action
            # unit and checkpoint convention belong to PRESS, not the provider.
            intervals = self.player_possessions_data.copy()
            if "possession_id" not in intervals:
                raise ValueError(
                    "Individual possessions require possession_id to construct canonical actions."
                )
            if "match_id" not in intervals:
                intervals["match_id"] = "" if match_id is None else str(match_id)
            if "game_interruption_after" not in intervals:
                intervals["game_interruption_after"] = None
            if "start_type" not in intervals:
                intervals["start_type"] = "unknown"
            actions_data = canonical_actions(intervals, fps=self.fps)
            if "outcome_verified" in intervals:
                actions_data["outcome_verified"] = actions_data.possession_id.map(
                    intervals.set_index("possession_id").outcome_verified
                )
        if actions_data is not None:
            self.actions_data = validate_intervals(actions_data, [
                "frame_start", "frame_end", "action", "start_type", "previous_action",
            ])
        self.results = {name: {} for name in (
            "PPCF", "BALL_DRIVE_RISK", "ACTION_SELECTION", "HAZARD", "BCOS", "RESIDUAL_RISK", "XLOSS",
        )}
        # Shared objects: result rows contain references, never copies of models.
        self.result_models = {}
        self.result_configurations = {}

    @property
    def fps(self):
        """Input frame frequency in Hz, independent of numerical solver steps."""
        return self.match_info.fps

    @staticmethod
    def _load_model(value, model_class):
        """Resolve a model, physical-parameter dictionary or exported manifest."""
        if value is None or isinstance(value, model_class):
            return value
        if isinstance(value, dict) and hasattr(model_class, 'from_parameters'):
            return model_class.from_parameters(value)
        if isinstance(value, (str, Path)):
            return model_class.from_json(value)
        raise TypeError(f"Expected {model_class.__name__}, a manifest path, or None.")

    def _prepare_roster(self, supplied):
        """Merge optional player metadata with IDs present in tracking."""
        import re

        ids = [c[:-2] for c in self.tracking_data if re.fullmatch(r"(home|away)_\d+_x", c)]
        roster = pd.DataFrame({"player_id": ids})
        roster["team_id"] = roster.player_id.str.split("_").str[0]
        if supplied is not None:
            if not isinstance(supplied, pd.DataFrame) or "player_id" not in supplied:
                raise ValueError("player_metadata must be a DataFrame keyed by player_id.")
            if supplied.player_id.isna().any() or supplied.player_id.duplicated().any():
                raise ValueError("player_metadata.player_id must be unique and nonmissing.")
            if not supplied.player_id.isin(ids).all():
                raise ValueError("Player metadata contains IDs absent from tracking.")
            if "team_id" in supplied:
                expected = supplied.player_id.str.split("_").str[0]
                if not supplied.team_id.eq(expected).all():
                    raise ValueError("Player metadata team_id conflicts with canonical IDs.")
            roster = roster.merge(supplied.drop(columns="team_id", errors="ignore"),
                                  on="player_id", how="left", validate="one_to_one")
        return roster

    @classmethod
    def from_skillcorner(cls, match_dir, *, match_id=None, fps=10.0, xT_grid=None, **models):
        """Load one original SkillCorner match and initialize PRESS.

        The folder contains <id>_match.json, <id>_tracking_extrapolated.jsonl
        and <id>_dynamic_events.csv. The folder name supplies id unless
        match_id is explicit. Velocities, annotated possessions, context and
        canonical actions are prepared in memory; no Parquet is required.
        Optional model arguments replace the bundled pretrained models.
        """
        from PRESS.io.skillcorner import load_skillcorner

        allowed = {'ball_drive_model', 'pass_risk_model', 'pass_selection_model',
                   'action_selection_model', 'residual_risk_model'}
        unknown = set(models)-allowed
        if unknown:
            raise TypeError('Unsupported SkillCorner loader arguments: '+str(sorted(unknown)))
        inputs, source_events, report = load_skillcorner(match_dir, match_id=match_id, fps=fps)
        press = cls(**inputs, xT_grid=xT_grid, **models)
        press.preprocessing_events = source_events
        press.data_preparation_report = report
        return press
