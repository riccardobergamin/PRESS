"""Export verified aggregate AS70/71/72 diagnostics, without fitting or weights."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT.parent/'skillcorner-analysis'
AS=RESEARCH/'experiments/70_action_selection_fine_bcos_20261029'
SOURCE=AS/'results_review/70_action_selection_fine_bcos_20261029_102502'
PROBE=RESEARCH/'experiments/71_action_selection_team_context_probe_20261029'
TEMP=RESEARCH/'experiments/72_action_selection_temperature_check_20261029/results'
OUT=ROOT/'data/benchmarks/action_selection'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    verification=json.loads((SOURCE/'verification.json').read_text())
    assert verification['status']=='passed' and verification['selected_models_replayed']
    summary=json.loads((SOURCE/'report/summary.json').read_text())
    temp=json.loads((TEMP/'summary.json').read_text());assert temp['status']=='complete_verified'
    OUT.mkdir(parents=True,exist_ok=True)
    exports={'metrics_by_fold.csv':SOURCE/'report/metrics_by_fold.csv',
             'metrics_by_match.csv':SOURCE/'report/metrics_by_match.csv',
             'summary.json':SOURCE/'report/summary.json',
             'historical_cohort_comparison.json':SOURCE/'historical_cohort_comparison.json',
             'context_probe_summary.csv':PROBE/'three_seed_summary.csv',
             'context_probe_by_seed.csv':PROBE/'three_seed_comparison.csv'}
    for name,source in exports.items():shutil.copy2(source,OUT/name)
    rows=[dict(model='native_raw',status='primary_frozen_benchmark',**temp['raw']),
          dict(model='temperature_diagnostic',status='post_development_diagnostic_only',**temp['temperature'])]
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    data=pd.read_parquet(TEMP/'diagnostic_oof.parquet');calibration=[]
    for column,label in [('probability','native_raw'),('temperature_probability','temperature_diagnostic')]:
        p=data[column].to_numpy(float);y=data.y.to_numpy(int)
        for i,ix in enumerate(np.array_split(np.argsort(p,kind='stable'),10)):
            calibration.append(dict(model=label,bin=i,n=len(ix),predicted=float(p[ix].mean()),observed=float(y[ix].mean())))
    bins=pd.DataFrame(calibration)
    for row in rows:
        b=bins.loc[bins.model.eq(row['model'])]
        assert abs((b.n*abs(b.predicted-b.observed)).sum()/16918-row['ece_quantile10'])<1e-12
    bins.to_csv(OUT/'calibration_deciles.csv',index=False)
    configurations=[];schemas=[]
    for f in range(5):
        s=json.loads((SOURCE/f'folds/fold_{f}/selection.json').read_text())
        spec=json.loads((SOURCE/f'folds/fold_{f}/feature_spec.json').read_text())
        configurations.append({k:s[k] for k in ['fold','max_depth','min_child_weight','reg_lambda','eta','best_round','trained_rounds','validation_ce','fit_ce_at_selection','fit_n','validation_n','test_n','reached_tree_cap']})
        schemas.append(dict(fold=f,**spec))
    pd.DataFrame(configurations).to_csv(OUT/'selected_configurations.csv',index=False)
    write(OUT/'feature_schemas.json',schemas)
    pd.DataFrame([dict(fold=f['fold'],temperature=f['temperature'],validation_ce_raw=f['validation_ce_raw'],validation_ce_scaled=f['validation_ce_scaled']) for f in temp['folds']]).to_csv(OUT/'temperature_diagnostic.csv',index=False)
    literature=[]
    for model,ce,brier in [('major_categories',.5762,.1982),('specific_types_included',.4807,.1617)]:
        literature.append(dict(study='Stalpaert2022',model=model,cross_entropy=ce,null_cross_entropy=.6602,
            gain_on_null=1-ce/.6602,brier=brier,locator='Section4.6, Table4.6, printed p39/PDF p49',
            caveat='Different dataset; specific preceding-action types are human annotated. Table4.7 ECE is postcalibration and not assigned to the Table4.6 variants.'))
    pd.DataFrame(literature).to_csv(OUT/'literature_comparison.csv',index=False)
    native=summary['pooled']
    for name in ['cross_entropy','gain_on_null','ece_quantile10']:assert abs(native[name]-temp['raw'][name])<1e-12
    sources=[SOURCE/'verification.json',SOURCE/'report/summary.json',SOURCE/'report/pooled_oof.parquet',
             SOURCE/'historical_cohort_comparison.json',AS/'bundle/data/protocol.json',
             PROBE/'conclusion.json',PROBE/'three_seed_summary.csv',TEMP/'summary.json',TEMP/'diagnostic_oof.parquet',
             RESEARCH/'tmp/pdfs/counterrisk.txt']
    installed=json.loads((ROOT/'PRESS/utils/data/action_selection_model.json').read_text())
    modern=installed.get('model_type')=='binary_action_choice'
    provenance=dict(status='verified_aggregate_export',source_run=SOURCE.name,n_actions=16918,n_pass=7770,n_carry=9148,
        primary_metrics=native,outer_folds=5,split='6 fitting + 2 validation + 2 held-out matches',
        calibration='Primary AS probabilities are native. Separate temperature diagnostic fits only the existing validation pair; it does not alter primary scores.',
        package_action_model_updated=modern,full_data_refit_performed=modern,
        deployment_status=('Verified full-ten-match native277-tree AS73 model is installed; AS70 remains the held-out benchmark.' if modern
            else 'Installed action_selection_model.json is historical; refit/runtime export remain separate.'),
        public_data='Only aggregate metrics, bins, configuration/schema and provenance; no raw tracking or individual prediction arrays exported.',
        development_caveat='Ten matches were used in prior model development. Upstream fitting-state scores are in-sample; same validation pair supports upstream calibration and action-model tuning.',
        source_sha256={str(s.relative_to(RESEARCH)):sha(s) for s in sources},
        export_sha256={f.name:sha(f) for f in OUT.iterdir() if f.is_file()})
    write(ROOT/'docs/action-selection-benchmark-provenance.json',provenance)
    print('Verified and exported action-selection aggregates; no model weights changed.')

if __name__=='__main__':main()
