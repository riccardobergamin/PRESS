"""Export arrival-dispersion uncertainty from frozen fitted artifacts; no refit."""
from pathlib import Path
import hashlib
import json
import math
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]


def main():
    bd_path = ROOT / 'PRESS/utils/data/ball_drive_model.json'
    pp_path = ROOT / 'PRESS/utils/data/pass_risk_model.json'
    bd = json.loads(bd_path.read_text())
    pp = json.loads(pp_path.read_text())
    if not bd['fit_diagnostics']['laplace_valid']:
        raise ValueError('The installed ball-drive covariance is not valid.')
    # theta[3] is log(s); take its marginal from the full joint covariance.
    mu = bd['theta'][3]
    variance = bd['posterior_covariance'][3][3]
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError('Invalid marginal variance for log(s).')
    sd = math.sqrt(variance)
    intervals = {}
    for coverage in (.68, .95):
        z = NormalDist().inv_cdf((1 + coverage) / 2)
        intervals[str(coverage)] = [math.exp(mu-z*sd), math.exp(mu+z*sd)]
    summary = {
        'method': 'Marginal joint-posterior Laplace normal on log(s), transformed to seconds',
        'ball_drive': {
            'model_id': bd['model_id'],
            'artifact_sha256': hashlib.sha256(bd_path.read_bytes()).hexdigest(),
            'parameter_index': 3,
            'parameter': 'log(s)',
            'transformed_mode_s_seconds': math.exp(mu),
            'log_s_mode': mu,
            'variance_log_s': variance,
            'sd_log_s': sd,
            'delta_method_sd_s_seconds': math.exp(mu)*sd,
            'equal_tailed_credible_intervals_seconds': intervals,
            'observations': bd['training']['observations'],
            'losses': bd['training']['losses'],
        },
        'pass_risk': {
            'model_id': pp['model_id'],
            'artifact_sha256': hashlib.sha256(pp_path.read_bytes()).hexdigest(),
            'mle_s_seconds': pp['s'],
            'uncertainty': 'No pass-risk credible interval inferred from the ball-drive covariance.',
        },
        'conditioning': 'Fixed kinematics, cohort, spline specification and selected prior/penalty strengths; no between-match dependence correction or model-selection/systematic uncertainty included.',
        'interpretation': 'Effective model-specific temporal dispersion; neither a universal physiological constant nor an independent validation across the two shared-data/shared-TTI models.',
        'historical_references': {
            'spearman2017_s_seconds': .45,
            'spearman2018_MAP_s_seconds': .54,
        },
    }
    out = ROOT / 'data/benchmarks/ball_drive/arrival_dispersion_laplace.json'
    out.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
