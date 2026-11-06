"""Copied result access, dependency invalidation and persistence for one match."""

from copy import deepcopy

import numpy as np


class ResultsMixin:
    """Internal implementation of the public PRESS result methods."""

    _COMPONENT_ALIASES = {
        "PASS_RISK": "PPCF", "PASS_SELECTION": "PPCF", "BALL_DRIVE": "BALL_DRIVE_RISK",
        "H": "HAZARD", "RESIDUAL": "RESIDUAL_RISK",
    }

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
                    self._invalidate(frame, "ACTION_SELECTION", "HAZARD")
                elif name == "BCOS":
                    self._invalidate(frame, "ACTION_SELECTION", "HAZARD")
                else:
                    self._invalidate(frame, name)

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
