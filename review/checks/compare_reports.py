"""Compare recomputed real-data reports/metrics with the published references."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from review.io import read_table, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output, inputs = args.output_root, ROOT / "review/inputs"
    states = read_table(output / "composed_risk/features/held_out.parquet")
    assert len(states) == 15085 and int(states.h_loss.sum()) == 1465 and int(states.terminal_loss.sum()) == 2888
    assert int(states.band_eligible.sum()) == 14919
    reference = read_table(inputs / "composed_risk/reference/components.parquet")
    joined = states.merge(reference, on=["match_id","action_id","frame_start"], suffixes=("", "_reference"), validate="one_to_one")
    # Original mixtures used native float32 q; recomposition promotes inputs
    # to float64. Preserve and report that rounding difference explicitly.
    errors = dict(h=float(abs(joined.h-joined.reference_h).max()),
                  xloss=float(abs(joined.xloss-joined.reference_xloss).max()))
    assert max(errors.values()) < np.finfo(np.float32).eps
    from_models = output / "composed_from_models/features/held_out.parquet"
    if from_models.is_file():
        predicted = read_table(from_models)
        paired = states.merge(predicted,on=["match_id","action_id","frame_start"],suffixes=("_saved","_model"),validate="one_to_one")
        assert len(paired)==15085
        for name in ["p_pass","residual_loss_probability","h","xloss"]:
            np.testing.assert_allclose(paired[name+"_saved"],paired[name+"_model"],atol=1e-7,rtol=0)
    performance = pd.read_csv(output / "composed_risk/scores/pooled_metrics.csv")
    published = pd.read_csv(inputs / "composed_risk/reference/pooled_metrics.csv")
    for model, name in [("h","h"),("xloss","xloss_with_action_choice")]:
        a = performance.loc[performance.model.eq(model)].iloc[0]
        b = published.loc[published.model.eq(name) & published.cohort.eq("verified_endpoints")].iloc[0]
        for column in ["cross_entropy","null_cross_entropy","gain_on_null","ece_quantile10"]:
            assert abs(a[column]-b[column]) < 2e-8, (model,column,a[column],b[column])
    for filename, keys in [("pattern_counts.csv",["pattern"]),("player_summary.csv",["provider_player_id"]),
                           ("team_summary.csv",["team_id"]),("team_peak_bands.csv",["team_name","peak_band"]),
                           ("endpoint_transitions.csv",["start_band","end_band"])]:
        new = pd.read_csv(output / "reports/scores" / filename)
        old = pd.read_csv(inputs / "reports/reference" / filename)
        assert len(new) == len(old)
        paired = new.merge(old,on=keys,suffixes=("_new","_old"),validate="one_to_one")
        for col in set(new.select_dtypes("number")) & set(old.select_dtypes("number")) - set(keys):
            np.testing.assert_allclose(paired[col+"_new"],paired[col+"_old"],atol=1e-6,rtol=0,equal_nan=True)
    bands = pd.read_csv(output / "composed_risk/bands/summary.csv",keep_default_na=False)
    published_bands = pd.read_csv(inputs / "composed_risk/reference/bands.csv",keep_default_na=False)
    paired = bands.merge(published_bands,on="band",suffixes=("_new","_old"),validate="one_to_one")
    np.testing.assert_array_equal(paired.n_new,paired.n_old)
    np.testing.assert_array_equal(paired.failures_new,paired.failures_old)
    for col in ["mean_h","observed_failure","mean_h_lower95","mean_h_upper95","gap_lower95","gap_upper95"]:
        np.testing.assert_allclose(paired[col+"_new"],paired[col+"_old"],atol=1e-8,rtol=0)
    write_json(output / "report_verification.json",dict(status="passed",states=len(states),
        band_states=14919,players=192,teams=12,maximum_composition_roundoff=errors,
        counts_identical=True,bootstrap_and_metrics_agree=True,new_training=False))


if __name__ == "__main__":
    main()
