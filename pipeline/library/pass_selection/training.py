"""Fixed in-process XGBoost utility paths and fit-only scale updates.

Keep live Booster objects across budget rungs: model serialization does not
restore all native stochastic state. Disk checkpoints are audit artifacts,
not permission to resume a stochastic trajectory in a different process.
"""
import json
from pathlib import Path
import time
import numpy as np
import xgboost as xgb
from pipeline.library.pass_selection.choice_likelihood import fit_rho
from pipeline.library.pass_selection.common import write

class Trajectory:

    def __init__(self, folder, params, fit_matrix, validation_matrix, fit_engine, validation_engine, protocol, restore=False):
        self.folder = Path(folder)
        self.params = params
        self.df, self.dv = (fit_matrix, validation_matrix)
        self.fit, self.val = (fit_engine, validation_engine)
        self.protocol = protocol
        assert not restore, 'Disk resume would not preserve native stochastic state.'
        self.folder.mkdir(parents=True, exist_ok=False)
        self.model = xgb.Booster(params=params, cache=[self.df, self.dv])
        self.state = dict(rounds=0, rho=1.0, best_round=0, best_rho=1.0, best_validation_ce=1e+99, stopped=False, history=[], updates=[], seconds=0.0, parameters=params, fit_history=[])
        assert self.model.num_boosted_rounds() == self.state['rounds']

    def advance(self, limit):
        start_time = time.monotonic()
        cfg = self.protocol['nesting']
        interval = cfg['update_every_trees']
        assert cfg['conditional_multistart'] is True
        while self.state['rounds'] < limit and (not self.state['stopped']):
            epoch = self.state['rounds']
            rho = self.state['rho']
            self.model.update(self.df, iteration=epoch, fobj=lambda scores, matrix: self.fit.gradient(scores, rho))
            step = epoch + 1
            if self.fit.mode == 'cnl' and step % interval == 0:
                scores = self.model.predict(self.df, output_margin=True)
                updated, record = fit_rho(self.fit, scores, rho, bounds=tuple(cfg['rho_bounds']), relaxation=cfg['relaxation'], scan=tuple(cfg['conditional_scan']))
                self.state['rho'] = updated
                record['after_tree'] = step
                self.state['updates'].append(record)
                print('SCALE', self.folder.name, step, json.dumps(record), flush=True)
            value = []

            def metric(scores, matrix):
                ce = float(self.val.losses(scores, self.state['rho']).mean())
                value.append(ce)
                return ('choice_nll', ce)
            self.model.eval_set([(self.dv, 'validation')], iteration=epoch, feval=metric, output_margin=True)
            assert len(value) == 1 and np.isfinite(value[0])
            ce = value[0]
            self.state['rounds'] = step
            self.state['history'].append(dict(round=step, rho=self.state['rho'], mu=1.0 / self.state['rho'], validation_ce=ce, negative_curvature_share=self.fit.last_negative_fraction, derivative_seconds=self.fit.last_seconds))
            if ce < self.state['best_validation_ce']:
                self.state.update(best_validation_ce=ce, best_round=step, best_rho=self.state['rho'])
            if step - self.state['best_round'] >= self.protocol['early_stopping_rounds']:
                self.state['stopped'] = True
            if step % self.protocol['fit_diagnostics_every_trees'] == 0 or self.state['stopped'] or step == limit:
                fit_ce = float(self.fit.losses(self.model.predict(self.df, output_margin=True), self.state['rho']).mean())
                assert np.isfinite(fit_ce)
                self.state['fit_history'].append(dict(round=step, fit_ce=fit_ce, validation_ce=ce, rho=self.state['rho']))
                print('TREE', self.folder.name, step, 'fit', fit_ce, 'validation', ce, 'best', self.state['best_validation_ce'], flush=True)
            if step % 100 == 0:
                self.save()
        self.state['seconds'] += time.monotonic() - start_time
        self.save()
        return self

    def save(self):
        assert self.model.num_boosted_rounds() == self.state['rounds']
        if self.state['rounds']:
            self.model.save_model(self.folder / 'last.ubj')
        write(self.folder / 'state.json', self.state)

    def selected(self):
        assert self.state['best_round'] > 0
        model = self.model[:self.state['best_round']]
        rho = self.state['best_rho']
        ce = float(self.val.losses(model.predict(self.dv, output_margin=True), rho).mean())
        assert np.isclose(ce, self.state['best_validation_ce'], atol=2e-06, rtol=0)
        return (model, rho)
