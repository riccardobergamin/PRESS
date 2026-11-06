"""Possession/frame navigation over explicit canonical tables."""


class NavigationMixin:
    """Internal implementation of the public PRESS methods."""

    def get_possessions(self, *, team_id=None):
        """Return a copy of explicit team intervals, optionally filtered by team.

        Raises ValueError when no team/player interval table was supplied.
        Carrier labels alone do not define verified team possessions.
        """
        if self.possessions_data is None:
            raise ValueError('Supply possessions_data or player_possessions_data to navigate possessions.')
        table = self.possessions_data
        if team_id is not None:
            if team_id not in ('home', 'away'):
                raise ValueError('team_id must be home or away.')
            table = table.loc[table.team_id == team_id]
        return table.copy().reset_index(drop=True)

    def get_possession(self, possession_id):
        """Return one team interval as a copied Series; unknown ID raises KeyError."""
        table = self.get_possessions()
        selected = table.loc[table.possession_id == possession_id]
        if selected.empty:
            raise KeyError(f'Unknown possession_id {possession_id!r}.')
        return selected.iloc[0].copy()

    def get_player_possessions(self, possession_id=None, *, team_id=None, player_id=None):
        """Return linked player intervals, optionally restricted to one team possession."""
        if self.player_possessions_data is None:
            raise ValueError('No player_possessions_data was supplied.')
        table = self.player_possessions_data
        if possession_id is not None:
            self.get_possession(possession_id)
            table = table.loc[table.team_possession_id == possession_id]
        if team_id is not None:
            if team_id not in ('home', 'away'):
                raise ValueError('team_id must be home or away.')
            table = table.loc[table.player_id.str.startswith(team_id + '_')]
        if player_id is not None:
            table = table.loc[table.player_id == player_id]
        return table.copy().reset_index(drop=True)

    def get_frame(self, frame_id):
        """Return one wide tracking row as a copied DataFrame; unknown ID raises KeyError."""
        frame = self.tracking_data.loc[self.tracking_data.frame == frame_id]
        if frame.empty:
            raise KeyError(f'Unknown frame_id {frame_id!r}.')
        return frame.copy().reset_index(drop=True)

    def get_frames(self, possession_id=None, *, valid_only=False):
        """Return sorted wide tracking rows, optionally within a team interval.

        Intervals include both endpoints and in-flight rows. valid_only
        excludes frames lacking a carrier, direction, or finite carrier
        position/velocity; it does not certify full tracking quality.
        """
        import numpy as np

        table = self.tracking_data
        if possession_id is not None:
            interval = self.get_possession(possession_id)
            table = table.loc[table.frame.between(interval.frame_start, interval.frame_end)]
        if valid_only:
            mask = table.player_possession.notna() & table.attacking_side.isin(['left_to_right', 'right_to_left'])
            for player in table.loc[mask, 'player_possession'].unique():
                selected = table.player_possession == player
                columns = [f'{player}_{suffix}' for suffix in ('x', 'y', 'vx', 'vy')]
                mask.loc[selected] &= np.isfinite(table.loc[selected, columns]).all(axis=1)
            table = table.loc[mask]
        return table.sort_values('frame').copy().reset_index(drop=True)

    def get_checkpoints(self, possession_id=None, *, eligible_only=True, include_snapshots=False):
        """Return canonical action starts, optionally with evaluation-only snapshots.

        Adds frame_id, start_type_std, previous_action_std, eligible.
        Canonical rows include drive attempts (also early failures) and pass
        releases. Short non-loss control tails and other terminal snapshots
        are optional and never action-selection training examples.
        Source possession_id, if present, remains a PLAYER-possession ID.
        """
        from PRESS.utils.match import standardize_action_context

        if self.actions_data is None:
            raise ValueError('Supply actions_data to navigate action checkpoints.')
        table = self.actions_data.copy()
        if possession_id is not None:
            interval = self.get_possession(possession_id)
            table = table.loc[table.frame_start.between(interval.frame_start, interval.frame_end)].copy()
        table['frame_id'] = table.frame_start
        if 'is_canonical' not in table:
            table['is_canonical'] = table.action.isin(['ball_drive', 'pass'])
        table['checkpoint_kind'] = table.is_canonical.map({True: 'canonical', False: 'snapshot'})
        table['start_type_std'] = table.start_type.map(standardize_action_context)
        table['previous_action_std'] = table.previous_action.map(
            lambda value: standardize_action_context(value, previous=True)
        )
        valid_frames = self.get_frames(possession_id, valid_only=True).frame
        table['eligible'] = (
            (
                table.is_canonical & table.action.isin(['ball_drive', 'pass'])
                | include_snapshots & table.action.eq('snapshot')
            )
            & table.frame_id.isin(valid_frames)
        )
        if eligible_only:
            table = table.loc[table.eligible]
            if table.frame_id.duplicated().any():
                raise ValueError('Ambiguous duplicate checkpoints; resolve actions_data first.')
        return table.reset_index(drop=True)

    def get_frame_context(self, frame_id, *, granular=None):
        """Read action context for any annotated tracking frame.

        Uses per-frame tracking context first, then an action interval
        containing this frame. At shared boundaries the latest interval
        start wins; conflicting contexts at that start raise ValueError.
        Default granular context retains unknown
        labels and includes completed_canonical_drives when recoverable from
        individual-control intervals at the configured input frequency or annotated action age. Set
        granular=False for descriptive grouped categories. Context
        extraction is independent of possession checkpoint selection.
        """
        if granular is None:
            granular = True
        if granular:
            return self._raw_action_context(frame_id)
        from PRESS.utils.match import standardize_action_context

        frame = self.get_frame(frame_id).iloc[0]
        for start_column, previous_column in (
            ('start_type_std', 'previous_action_std'),
            ('start_type', 'previous_action'),
        ):
            if start_column in frame and previous_column in frame:
                start = standardize_action_context(frame[start_column])
                previous = standardize_action_context(frame[previous_column], previous=True)
                if start is not None and previous is not None:
                    return {'start_type': start, 'previous_action': previous, 'source': 'tracking'}
        if self.actions_data is not None:
            intervals = self.actions_data.loc[
                (self.actions_data.frame_start <= frame_id)
                & (self.actions_data.frame_end >= frame_id)
            ]
            if not intervals.empty:
                intervals = intervals.loc[intervals.frame_start == intervals.frame_start.max()]
                contexts = {(
                    standardize_action_context(row.start_type),
                    standardize_action_context(row.previous_action, previous=True),
                ) for row in intervals.itertuples(index=False)}
                if len(contexts) != 1:
                    raise ValueError('Conflicting action context for this frame; supply context explicitly.')
                start, previous = contexts.pop()
                return {
                    'start_type': start,
                    'previous_action': previous,
                    'source': 'action_interval',
                }
        raise ValueError(
            'No action context annotation covers this frame; supply start_type '
            'and previous_action explicitly.'
        )

    def _raw_action_context(self, frame_id):
        """Resolve raw origin and integer past carry count at any usable frame."""
        import numpy as np
        import pandas as pd

        frame = self.get_frame(frame_id).iloc[0]
        context = dict(
            start_type=frame.get('start_type_raw', frame.get('start_type', None)),
            previous_action=frame.get('previous_action_raw', frame.get('previous_action', None)),
            completed_canonical_drives=frame.get('completed_canonical_drives', np.nan),
            time_since_start=frame.get('time_since_start', np.nan),
            source='tracking_or_missing',
        )
        if self.player_possessions_data is not None:
            candidates = self.player_possessions_data.loc[
                (self.player_possessions_data.frame_start <= frame_id)
                & (self.player_possessions_data.frame_end >= frame_id)
            ]
            if len(candidates) > 1:
                raise ValueError('Conflicting individual-control intervals.')
            if len(candidates):
                parent = candidates.iloc[0]
                elapsed = (frame_id - int(parent.frame_start)) / self.fps
                count = int(np.floor(elapsed + 1e-08))
                context.update(
                    start_type=parent.get('start_type', None),
                    previous_action='ball_drive' if count >= 1 else parent.get('start_type', None),
                    completed_canonical_drives=count,
                    time_since_start=float(elapsed),
                    source='individual_possession',
                )
                return context
        if self.actions_data is not None:
            intervals = self.actions_data.loc[
                (self.actions_data.frame_start <= frame_id)
                & (self.actions_data.frame_end >= frame_id)
            ]
            if len(intervals):
                intervals = intervals.loc[intervals.frame_start == intervals.frame_start.max()]
                if len(intervals[['start_type', 'previous_action']].drop_duplicates()) > 1:
                    raise ValueError('Conflicting raw action context.')
                row = intervals.iloc[0]
                context.update(
                    start_type=row.start_type,
                    previous_action=row.previous_action,
                    source='action_interval',
                )
                if pd.notna(row.get('time_since_start', np.nan)):
                    initial = float(row.frame_start) - self.fps * float(row.time_since_start)
                    elapsed = (frame_id - initial) / self.fps
                    count = int(np.floor(elapsed + 1e-08))
                    context.update(
                        completed_canonical_drives=count,
                        time_since_start=float(elapsed),
                        previous_action='ball_drive' if count >= 1 else row.start_type,
                    )
        return context

    def get_state(self, frame_id):
        """
        Extract physical state from frame.
        """
        from PRESS.utils.state import get_pv_players

        return get_pv_players(frame_id, self.tracking_data)

    def get_match_info(self):
        """Return input frequency, pitch dimensions and optional match ID."""
        return self.match_info.to_dict()

    def get_players(self, *, team_id=None):
        """Return a copied roster, including source IDs when supplied by an adapter."""
        table = self.player_metadata
        if team_id is not None:
            if team_id not in ('home', 'away'):
                raise ValueError('team_id must be home or away.')
            table = table.loc[table.team_id == team_id]
        return table.copy().reset_index(drop=True)

    def get_player_possession(self, player_possession_id):
        """Return one individual interval by its source possession_id.

        This differs from get_possession(), which selects a TEAM interval.
        No individual interval is inferred from changes of carrier labels.
        """
        table = self.get_player_possessions()
        if 'possession_id' not in table:
            raise ValueError('Individual intervals require possession_id for ID-based selection.')
        selected = table.loc[table.possession_id == player_possession_id]
        if selected.empty:
            raise KeyError(f'Unknown player_possession_id {player_possession_id!r}.')
        if len(selected) != 1:
            raise ValueError('Individual possession IDs must be unique within a match.')
        return selected.iloc[0].copy()

    def get_player_frames(self, player_possession_id, *, valid_only=False):
        """Return copied tracking rows inside an individual possession interval."""
        interval = self.get_player_possession(player_possession_id)
        frames = self.get_frames(valid_only=valid_only)
        return frames.loc[frames.frame.between(interval.frame_start, interval.frame_end)].reset_index(drop=True)

    def get_elapsed_time(self, frame_id):
        """Return observed individual-possession age in seconds.

        Uses a covering individual interval, or current/action context with
        a known elapsed time. Missing age raises ValueError; zero is never
        imputed. Input frame frequency is self.fps.
        """
        import numpy as np

        value = self._raw_action_context(frame_id).get('time_since_start', np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                'No elapsed individual-possession time is available; supply '
                'time_since_start explicitly.'
            ) from None
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                'No elapsed individual-possession time is available; supply '
                'time_since_start explicitly.'
            )
        return float(value)
