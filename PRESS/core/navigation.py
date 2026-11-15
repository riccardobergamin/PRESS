"""Possession/frame navigation over explicit canonical tables."""


from copy import deepcopy


class NavigationMixin:
    """Internal implementation of the public PRESS methods."""

    def get_team_possessions(self, possession_id=None, *, team_id=None):
        """Return team intervals, optionally selected by their ID and team.

        Always returns a copied DataFrame, including for a single ID.
        Omitted filters select all intervals; unmatched filters select none.
        Raises ValueError when no team/player interval table was supplied.
        Carrier labels alone do not define verified team possessions.
        """
        if self.possessions_data is None:
            raise ValueError('Supply possessions_data or player_possessions_data to navigate possessions.')
        table = self.possessions_data
        if possession_id is not None:
            table = table.loc[table.possession_id.eq(possession_id)]
        if team_id is not None:
            if team_id not in ('home', 'away'):
                raise ValueError('team_id must be home or away.')
            table = table.loc[table.team_id == team_id]
        return table.copy().reset_index(drop=True)

    def get_player_possessions(self, possession_id=None, *, team_id=None,
                               player_id=None, team_possession_id=None):
        """Select individual controls by their ID, team, player or team interval.

        No filters returns all individual possessions. possession_id always
        identifies an INDIVIDUAL control; team_possession_id is explicitly
        separate. Combined filters are intersected. Return a copied DataFrame.
        """
        if self.player_possessions_data is None:
            raise ValueError('No player_possessions_data was supplied.')
        table = self.player_possessions_data
        if possession_id is not None:
            if 'possession_id' not in table:
                raise ValueError('Individual intervals require possession_id for ID-based selection.')
            table = table.loc[table.possession_id.eq(possession_id)]
        if team_possession_id is not None:
            table = table.loc[table.team_possession_id.eq(team_possession_id)]
        if team_id is not None:
            if team_id not in ('home', 'away'):
                raise ValueError('team_id must be home or away.')
            table = table.loc[table.player_id.str.startswith(team_id + '_')]
        if player_id is not None:
            table = table.loc[table.player_id.eq(player_id)]
        return table.copy().reset_index(drop=True)

    def get_actions(self, possession_id=None, *, team_id=None, player_id=None,
                    team_possession_id=None, include_snapshots=False,
                    valid_only=False):
        """Select actions by individual possession, team, player or team interval.

        By default returns all canonical pass/carry actions. Set
        include_snapshots=True to include exposure/terminal snapshots.
        Each canonical action's frame_start is its checkpoint. valid_only
        retains starts with a usable carrier, attack direction and kinematics;
        it does not verify the observed outcome or run any model.
        Supplied action identifiers are preserved. Missing identifiers are
        recovered from explicit individual intervals, never carrier changes.
        Player identity can also be read from the action's tracking frame.
        """
        import numpy as np

        if self.actions_data is None:
            raise ValueError('Supply player_possessions_data or actions_data to navigate actions and checkpoints.')
        table = self.actions_data.copy()
        parents = self.player_possessions_data
        # Attach optional identity columns without mutating actions_data.
        # Inclusive, nonoverlapping parent intervals uniquely identify each
        # action start; this also supports explicitly supplied minimal tables.
        if parents is not None and len(parents):
            parents = parents.sort_values('frame_start', kind='stable')
            indices = np.searchsorted(parents.frame_start.to_numpy(),
                                      table.frame_start.to_numpy(), side='right')-1
            covered = indices >= 0
            covered[covered] &= (table.frame_start.to_numpy()[covered]
                                <= parents.frame_end.to_numpy()[indices[covered]])
            for column in ['possession_id', 'player_id', 'team_possession_id']:
                if column not in table and column in parents:
                    values = np.full(len(table), None, dtype=object)
                    values[covered] = parents[column].to_numpy(dtype=object)[indices[covered]]
                    table[column] = values
        if 'player_id' not in table:
            carriers = self.tracking_data.set_index('frame').player_possession
            table['player_id'] = table.frame_start.map(carriers)
        if 'team_id' not in table:
            table['team_id'] = table.player_id.astype('string').str.split('_').str[0]
        if 'is_canonical' not in table:
            table['is_canonical'] = table.action.isin(['ball_drive', 'pass'])
        if not include_snapshots:
            table = table.loc[table.is_canonical]
        if possession_id is not None:
            if 'possession_id' not in table:
                raise ValueError('Individual selection requires actions_data.possession_id or explicit individual intervals.')
            table = table.loc[table.possession_id.eq(possession_id)]
        if team_possession_id is not None:
            interval = self.get_team_possessions(team_possession_id)
            if interval.empty:
                return table.head(0).copy().reset_index(drop=True)
            table = table.loc[table.frame_start.between(interval.frame_start.item(), interval.frame_end.item())]
        if team_id is not None:
            if team_id not in ('home', 'away'):
                raise ValueError('team_id must be home or away.')
            table = table.loc[table.team_id.eq(team_id)]
        if player_id is not None:
            table = table.loc[table.player_id.eq(player_id)]
        if valid_only:
            valid_frames = self.get_frames(team_possession_id, valid_only=True).frame
            kinds = ['ball_drive', 'pass'] + (['snapshot'] if include_snapshots else [])
            table = table.loc[table.action.isin(kinds) & table.frame_start.isin(valid_frames)]
            if table.frame_start.duplicated().any():
                raise ValueError('Ambiguous duplicate action starts; resolve actions_data first.')
        return table.copy().reset_index(drop=True)

    def get_frame(self, frame_id):
        """Return one wide tracking row as a copied DataFrame; unknown ID raises KeyError."""
        cache = getattr(self, '_frame_inputs', None)
        if cache is not None and cache['frame_id'] == frame_id and 'frame' in cache:
            return cache['frame'].copy().reset_index(drop=True)
        frame = self.tracking_data.loc[self.tracking_data.frame == frame_id]
        if frame.empty:
            raise KeyError(f'Unknown frame_id {frame_id!r}.')
        if cache is not None and cache['frame_id'] == frame_id:
            cache['frame'] = frame
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
            interval = self.get_team_possessions(possession_id)
            if interval.empty:
                raise KeyError(f'Unknown team possession_id {possession_id!r}.')
            table = table.loc[table.frame.between(interval.frame_start.item(), interval.frame_end.item())]
        if valid_only:
            mask = table.player_possession.notna() & table.attacking_side.isin(['left_to_right', 'right_to_left'])
            for player in table.loc[mask, 'player_possession'].unique():
                selected = table.player_possession == player
                columns = [f'{player}_{suffix}' for suffix in ('x', 'y', 'vx', 'vy')]
                mask.loc[selected] &= np.isfinite(table.loc[selected, columns]).all(axis=1)
            table = table.loc[mask]
        return table.sort_values('frame').copy().reset_index(drop=True)

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

        cache = getattr(self, '_frame_inputs', None)
        if cache is not None and cache['frame_id'] == frame_id and 'context' in cache:
            return dict(cache['context'])
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
                if cache is not None and cache['frame_id'] == frame_id:
                    cache['context'] = dict(context)
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
        if cache is not None and cache['frame_id'] == frame_id:
            cache['context'] = dict(context)
        return context

    def get_state(self, frame_id):
        """
        Extract physical state from frame.
        """
        from PRESS.utils.state import get_pv_players

        cache = getattr(self, '_frame_inputs', None)
        if cache is not None and cache['frame_id'] == frame_id:
            if 'state' not in cache:
                cache['state'] = get_pv_players(frame_id, self.get_frame(frame_id))
            return deepcopy(cache['state'])
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

    def get_player_frames(self, player_possession_id, *, valid_only=False):
        """Return copied tracking rows inside an individual possession interval."""
        interval = self.get_player_possessions(possession_id=player_possession_id)
        if interval.empty:
            raise KeyError(f'Unknown player_possession_id {player_possession_id!r}.')
        if len(interval) != 1:
            raise ValueError('Individual possession IDs must be unique within a match.')
        frames = self.get_frames(valid_only=valid_only)
        return frames.loc[frames.frame.between(interval.frame_start.item(), interval.frame_end.item())].reset_index(drop=True)

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


    def get_preparation_report(self):
        """Return the raw-loader preparation report, or None for direct table input."""
        from copy import deepcopy
        return deepcopy(getattr(self, 'data_preparation_report', None))

    def get_preprocessing_events(self):
        """Return original-loader event audit records, or None for direct table input."""
        events = getattr(self, 'preprocessing_events', None)
        return None if events is None else events.copy(deep=True)


    def get_xt_grid(self, frame_id=None):
        """Return a copied xT map; optional frame selects the map actually used."""
        if frame_id is not None:
            field = self.get_ppcf_result(frame_id)
            if 'configuration_id' in field:
                configuration = self.result_configurations[field['configuration_id']]
                if 'xT grid' not in configuration:
                    raise ValueError('Compute pass selection to identify the frame xT map.')
                return configuration['xT grid'].copy()
            if 'xT grid' not in field:
                raise ValueError('Compute pass selection to identify the frame xT map.')
            return field['xT grid'].copy()
        return self._resolve_xt_grid(None)
