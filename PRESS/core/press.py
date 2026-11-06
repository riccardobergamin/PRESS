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

    Use canonical wide tracking directly, ``from_long`` for long tables from
    another provider, or ``from_skillcorner`` for the project's processed
    files. Initialization validates/copies inputs; it runs no predictions.
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
            action, start_type, previous_action``. No segmentation is guessed.
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
            included. ``from_long`` constructs the source-ID mapping itself.
        ball_drive_model, pass_risk_model, pass_selection_model,
        action_selection_model, residual_risk_model : model or path, optional
            Explicit fitted model or JSON-manifest path. Bundled production
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
        if actions_data is not None:
            self.actions_data = validate_intervals(actions_data, [
                "frame_start", "frame_end", "action", "start_type", "previous_action",
            ])
        self.results = {name: {} for name in (
            "PPCF", "BALL_DRIVE_RISK", "ACTION_SELECTION", "HAZARD", "BCOS", "RESIDUAL_RISK", "XLOSS",
        )}

    @property
    def fps(self):
        """Input frame frequency in Hz, independent of numerical solver steps."""
        return self.match_info.fps

    @staticmethod
    def _load_model(value, model_class):
        """Validate a model instance or load its exported manifest."""
        if value is None or isinstance(value, model_class):
            return value
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
    def from_long(cls, players, ball, context, *, home_team="home", away_team="away", **kwargs):
        """Initialize from provider-independent long player/ball/context tables.

        See ``PRESS.io.tracking_from_long`` for the three table schemas.
        Source IDs are mapped to internal home/away aliases and retained by
        ``get_players``. Optional player/event/team interval tables in kwargs
        use source IDs too; they are remapped without inferring outcomes.
        Rename source columns and convert coordinates/units before this call.
        """
        from PRESS.io import tracking_from_long

        tracking, roster = tracking_from_long(
            players, ball, context, home_team=home_team, away_team=away_team,
        )
        teams = {home_team: "home", away_team: "away"}
        sources = roster.groupby("source_player_id", sort=False).player_id.agg(list).to_dict()
        pairs = {(r.source_team_id, r.source_player_id): r.player_id for r in roster.itertuples()}
        for name in ("player_possessions_data", "events_data", "possessions_data", "actions_data"):
            if kwargs.get(name) is None:
                continue
            table = kwargs[name].copy()
            for column in ("player_id", "target_pass", "receiver_player_id"):
                if column not in table:
                    continue
                mapped = []
                for _, row in table.iterrows():
                    value = row[column]
                    if pd.isna(value):
                        mapped.append(None)
                        continue
                    alias = pairs.get((row.get("team_id"), value))
                    candidates = sources.get(value, [])
                    if alias is None and len(candidates) == 1:
                        alias = candidates[0]
                    if alias is None:
                        raise ValueError(f"Unknown or ambiguous source ID in {name}.{column}.")
                    mapped.append(alias)
                table[column] = mapped
            if "team_id" in table:
                if not table.team_id.isin(teams).all():
                    raise ValueError(f"Unknown source team in {name}.")
                table["team_id"] = table.team_id.map(teams)
            kwargs[name] = table
        if "player_metadata" in kwargs:
            raise ValueError("from_long builds player_metadata from the source roster.")
        return cls(tracking, player_metadata=roster, **kwargs)

    @classmethod
    def from_skillcorner(cls, match_dir, *, actions_path=None, **kwargs):
        """Load the project's processed SkillCorner parquet files at 10 Hz.

        The directory contains ``<match_id>_tracking.parquet`` and
        ``<match_id>_possession.parquet``. Optional multi-match actions are
        filtered by match_id. This is a convenience file adapter, not a raw
        SkillCorner preprocessing pipeline. Requires the ``io`` extra.
        """
        match_dir = Path(match_dir)
        match_id = match_dir.name
        tracking = pd.read_parquet(match_dir / f"{match_id}_tracking.parquet")
        players = pd.read_parquet(match_dir / f"{match_id}_possession.parquet")
        actions = None
        if actions_path is not None:
            actions_path = Path(actions_path)
            actions = (pd.read_parquet(actions_path) if actions_path.suffix == ".parquet"
                       else pd.read_csv(actions_path, dtype={"match_id": str}))
            if "match_id" not in actions:
                raise ValueError("Actions must contain match_id.")
            actions = actions.loc[actions.match_id.astype(str) == match_id].copy()
        kwargs.setdefault("match_id", match_id)
        return cls(tracking, player_possessions_data=players, actions_data=actions, **kwargs)
