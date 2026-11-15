"""Copied result access, dependency invalidation and persistence for one match."""

from copy import deepcopy
import hashlib
import json

import numpy as np


class ResultsMixin:
    """Internal implementation of the public PRESS result methods."""

    _COMPONENT_ALIASES = {
        "PASS_RISK": "PPCF", "PASS_SELECTION": "PPCF", "BALL_DRIVE": "BALL_DRIVE_RISK",
        "H": "HAZARD", "RESIDUAL": "RESIDUAL_RISK",
    }

    def _register_result_model(self, model):
        """Retain one model per fitted artifact, including per-call overrides.

        Use an artifact digest, not its human-readable name: two custom
        parameter sets may legitimately have the same model_id.
        """
        for key, known in self.result_models.items():
            if known is model:
                return key
        artifact = json.dumps(model.to_dict(), sort_keys=True, allow_nan=False)
        key = hashlib.sha256(artifact.encode()).hexdigest()
        self.result_models.setdefault(key, model)
        return key

    def _store_result_configuration(self, configuration):
        """Keep one copy of resolved settings and return its content identifier."""
        text = json.dumps(configuration, sort_keys=True,
                          default=lambda value: np.asarray(value).tolist())
        key = hashlib.sha256(text.encode()).hexdigest()
        self.result_configurations.setdefault(key, deepcopy(configuration))
        return key

    def _record_component_configuration(self, component, frame_id, settings,
                                        *, models=None, xt_grid=None):
        """Attach shared settings to one component, including standalone calls.

        Only execution settings go here. Estimated coefficients, spline bases
        and calibration remain in the referenced fitted models.
        """
        result = self.results[component][frame_id]
        references = dict(models or {})
        if 'model_ref' in result:
            references[component] = result['model_ref']
        configuration = {'models': references, component: dict(settings)}
        if xt_grid is not None:
            configuration['xT grid'] = np.asarray(xt_grid).copy()
        result['configuration_id'] = self._store_result_configuration(configuration)
        result['details_available'] = True

    def _frame_configuration(self, frame_id):
        """Collect settings only from the components currently valid at this frame.

        A stored configuration can describe a complete calculation. Select only
        the relevant component from it: an upstream recomputation must not bring
        back the settings of an invalidated downstream prediction.
        """
        configuration = {'models': {}}
        for name, frames in self.results.items():
            result = frames.get(frame_id)
            if result is None or 'configuration_id' not in result:
                continue
            source = self.result_configurations[result['configuration_id']]
            if name in source:
                configuration[name] = source[name]
            reference = result.get('model_ref', source['models'].get(name))
            if reference is not None:
                configuration['models'][name] = reference
            if name == 'PPCF' and any(field in result for field in
                    ['Pass selection probabilities', 'Expected pass success']):
                choice_ref = result.get('pass_selection_model_ref', source['models'].get('PASS_SELECTION'))
                if choice_ref is not None:
                    configuration['models']['PASS_SELECTION'] = choice_ref
                if 'xT grid' in source:
                    configuration['xT grid'] = source['xT grid']
            if name == 'RESIDUAL_RISK' and 'ACTION_SELECTION' in source['models']:
                configuration['models']['ACTION_SELECTION'] = source['models']['ACTION_SELECTION']
        return configuration

    def _finalize_frame_results(self, frame_id, xt_grid, *, keep_details):
        """Discard intermediates only AFTER all requested predictions finish.

        Standalone calls already recorded their configurations. A complete call
        combines those references and applies its requested storage level.
        """
        available = {name: frames[frame_id] for name, frames in self.results.items()
                     if frame_id in frames}
        configuration = self._frame_configuration(frame_id)
        configuration['xT grid'] = np.asarray(xt_grid).copy()
        key = self._store_result_configuration(configuration)

        # Destination-weighted summaries remain useful without any spatial map.
        hazard = available['HAZARD']
        ppcf = available['PPCF']
        ppcf['Expected pass success'] = hazard['rppcf']
        ppcf['Expected pass failure'] = hazard['Pass failure expected']
        # Context can be overridden by the caller. Preserve those few inputs
        # per frame even when the full derived feature record is discarded.
        ppcf['context'] = dict(ppcf.get('pass_selection_context', {}))
        action = available['ACTION_SELECTION']
        features = action.get('Features', {})
        action['context'] = {field: features[field] for field in [
            'start_type_raw', 'previous_action_raw', 'match_clock_min',
            'score_difference_prefix', 'completed_canonical_drives',
        ] if field in features}
        scalar_fields = {
            'PPCF': ['Expected pass success', 'Expected pass failure', 'context'],
            'BALL_DRIVE_RISK': ['Ball drive risk', 'Survival probability'],
            'ACTION_SELECTION': ['Pass probability', 'Ball drive probability',
                                 'BCOS feature formulation', 'context'],
            'HAZARD': ['Hazard', 'band', 'Drive risk', 'Pass failure expected', 'rppcf'],
            'BCOS': ['BCOS area', 'Model feature BCOS area'],
            'RESIDUAL_RISK': ['Residual loss probability', 'time_since_start'],
            'XLOSS': ['xLoss', 'Hazard', 'Residual loss probability', 'time_since_start'],
        }
        for name, result in available.items():
            if not keep_details:
                result = {field: result[field] for field in scalar_fields[name] if field in result}
            result['configuration_id'] = key
            result['details_available'] = keep_details
            self.results[name][frame_id] = result

        self._prune_result_settings()

    def get_result_configuration(self, frame_id, component=None):
        """Return copied execution settings for this frame or one component.

        Works after standalone or complete calculations. Omit component to
        collect the settings of all currently valid components. No inference
        is triggered and no model parameters are copied into the settings.
        """
        if component is not None:
            result = self.get_result(component, frame_id, copy=False)
            key = result.get('configuration_id')
            if key is None:
                raise ValueError('No shared configuration recorded for this result.')
            source = self._frame_configuration(frame_id)
            name = self._component(component)
            configuration = {'models': {}, name: source.get(name, {})}
            for model_name in [name] + (['PASS_SELECTION'] if name == 'PPCF' else []):
                if model_name in source['models']:
                    configuration['models'][model_name] = source['models'][model_name]
            if name == 'PPCF' and 'xT grid' in source:
                configuration['xT grid'] = source['xT grid']
        else:
            if not any(frame_id in frames for frames in self.results.values()):
                raise KeyError(f'No results stored for frame {frame_id!r}.')
            configuration = self._frame_configuration(frame_id)
        return deepcopy(configuration)

    def get_result_model(self, component, frame_id):
        """Return the fitted model actually used, including custom parameters.

        This is the shared model instance, not a per-frame copy. Inspect its
        parameters/bases or call to_dict() for a separate editable artifact.
        PASS_SELECTION selects the destination model; PPCF/PASS_RISK the
        physical completion model.
        """
        name = str(component).upper()
        component = 'PASS_SELECTION' if name == 'PASS_SELECTION' else self._component(name)
        result = self.get_result(self._component(component), frame_id, copy=False)
        if component == 'PASS_SELECTION' and not any(field in result for field in
                ['Pass selection probabilities', 'Expected pass success']):
            raise ValueError('No valid pass-selection result; compute pass selection first.')
        direct_key = 'pass_selection_model_ref' if component == 'PASS_SELECTION' else 'model_ref'
        key = result.get(direct_key)
        if key is None and 'configuration_id' in result:
            config = self.result_configurations[result['configuration_id']]
            key = config['models'].get(component)
        if key is None:
            raise ValueError(f'No fitted model reference recorded for {component} at frame {frame_id}.')
        return self.result_models[key]

    def _component(self, name):
        """Resolve public aliases to a stored result component."""
        name = str(name).upper()
        name = self._COMPONENT_ALIASES.get(name, name)
        if name not in self.results:
            raise ValueError(f"Unknown component {name!r}; expected one of {', '.join(self.results)}.")
        return name

    def _invalidate(self, frame_id, *components):
        """Invalidate downstream predictions when an upstream input changes."""
        expanded = set(components)
        if expanded & {"ACTION_SELECTION", "HAZARD", "BCOS"}:
            expanded.update(["RESIDUAL_RISK", "XLOSS"])
        if "ACTION_SELECTION" in expanded:
            expanded.add("HAZARD")
        if "RESIDUAL_RISK" in expanded:
            expanded.add("XLOSS")
        for component in expanded:
            self.results[component].pop(frame_id, None)

    def get_result(self, component, frame_id, *, copy=True):
        """Return a frame's result mapping, copied by default.

        Components are PPCF, BALL_DRIVE_RISK, ACTION_SELECTION, HAZARD, BCOS,
        RESIDUAL_RISK and XLOSS. PASS_RISK/PASS_SELECTION alias PPCF,
        BALL_DRIVE aliases BALL_DRIVE_RISK. Missing calculations raise KeyError.
        Set copy=False only when intentionally accessing the internal cache.
        """
        component = self._component(component)
        if frame_id not in self.results[component]:
            raise KeyError(f"No {component} result stored for frame {frame_id!r}.")
        value = self.results[component][frame_id]
        return deepcopy(value) if copy else value

    def get_results(self, component=None):
        """Return a copied component cache, or all computed component caches."""
        return (
            deepcopy(self.results
            if component is None
            else self.results[self._component(component)])
        )

    def get_ppcf_result(self, frame_id):
        """Return candidate grid, raw control and calibrated pass completion."""
        return self.get_result("PPCF", frame_id)

    def get_pass_selection_result(self, frame_id):
        """Return the destination grid and its stored conditional choice distribution."""
        result = self.get_ppcf_result(frame_id)
        if "Pass selection probabilities" not in result:
            if result.get('details_available') is False:
                raise ValueError('Spatial details were not retained; rerun compute_risk_frame with keep_details=True.')
            raise KeyError(f"No pass-selection result stored for frame {frame_id!r}.")
        return {key: result[key] for key in [
            "grid",
            "Pass selection probabilities",
            "pass_selection_context",
        ] if key in result}

    def get_ball_drive_risk_result(self, frame_id):
        """Return total and defender-specific ball-drive loss probabilities."""
        return self.get_result("BALL_DRIVE_RISK", frame_id)

    def get_action_selection_result(self, frame_id):
        """Return native P(pass) and P(carry) for a computed frame."""
        return self.get_result("ACTION_SELECTION", frame_id)

    def get_hazard_result(self, frame_id):
        """Return marginal next-action loss h and its branch quantities."""
        return self.get_result("HAZARD", frame_id)

    def get_bcos_result(self, frame_id):
        """Return the optional BCOS spatial field and its area in square metres."""
        return self.get_result("BCOS", frame_id)

    def get_residual_risk_result(self, frame_id):
        """Return conditional terminal loss after a successful one-second carry."""
        return self.get_result("RESIDUAL_RISK", frame_id)

    def get_xloss_result(self, frame_id):
        """Return the composed probability of terminal individual-possession loss."""
        return self.get_result("XLOSS", frame_id)

    def clear_results(self, component=None, *, frame_ids=None):
        """Clear cached results and their dependants, leaving input tables intact.

        Omitting component clears every component. Omitting frame_ids clears
        every stored frame. Clearing PPCF or BALL_DRIVE_RISK also removes stored pass
        choice and every downstream prediction for those frames.
        """
        components = list(self.results) if component is None else [self._component(component)]
        if frame_ids is None:
            frames = set().union(*(set(self.results[c]) for c in self.results))
        else:
            frames = list(frame_ids)
        for frame in frames:
            for name in components:
                self.results[name].pop(frame, None)
                if name in ("PPCF", "BALL_DRIVE_RISK"):
                    self.results["PPCF"].get(frame, {}).pop("Pass selection probabilities", None)
                    self.results["PPCF"].get(frame, {}).pop("Expected pass success", None)
                    self.results["PPCF"].get(frame, {}).pop("Expected pass failure", None)
                    self._invalidate(frame, "ACTION_SELECTION", "HAZARD")
                elif name == "BCOS":
                    self._invalidate(frame, "ACTION_SELECTION", "HAZARD")
                else:
                    self._invalidate(frame, name)
        # Release configurations/models no longer referenced by any result.
        self._prune_result_settings()

    def _prune_result_settings(self):
        """Keep settings only while at least one saved result references them."""
        configurations = set()
        models = set()
        for frames in self.results.values():
            for result in frames.values():
                if 'configuration_id' in result:
                    configurations.add(result['configuration_id'])
                models.update(result[key] for key in ['model_ref', 'pass_selection_model_ref'] if key in result)
        for key in list(self.result_configurations):
            if key not in configurations:
                del self.result_configurations[key]
            else:
                models.update(self.result_configurations[key]['models'].values())
        for key in list(self.result_models):
            if key not in models:
                del self.result_models[key]

    def save_results(self, path, *, overwrite=False):
        """Save computed results to a directory containing JSON and compressed NPZ.

        Arrays retain shapes/dtypes; no pickle, model or tracking table is
        serialized. The manifest identifies the input match/frequency and
        includes a tracking fingerprint. Existing exports require overwrite.
        Returns the export directory as pathlib.Path.
        """
        from PRESS.io.results import save_results

        return save_results(self, path, overwrite=overwrite)

    def load_results(self, path, *, overwrite=False):
        """Restore an export only when it matches this object's tracking and metadata.

        This loads numerical results, not model weights or a saved PRESS
        object. Existing frame/component results require overwrite=True.
        The complete export is validated before any cache is changed.
        """
        from PRESS.io.results import load_results

        return load_results(self, path, overwrite=overwrite)
