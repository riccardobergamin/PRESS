"""xT event/basis, leave-one-match-out smoothing, score and export checks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"review/expected_threat"))
import model
import training
import pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    inputs=ROOT/"review/inputs/expected_threat"
    events,protocol=pipeline.load_event_tables(inputs)
    binned=model.bin_events(events,(10,6))
    # One actual six-match fit exercises all three LOMO bandwidth selections.
    fitting=protocol["feature_fitting_matches"]["0"]
    fitted,bandwidths,scores,comparison=training.fit_on_matches(binned,fitting,(10,6))
    training.save_fit(args.output/"fold_0",fitted,(10,6),fitting,bandwidths,scores,comparison)
    reference=np.load(inputs/"reference/six_match_maps/fold_0/xt.npy")
    np.testing.assert_allclose(fitted["xt"],reference,atol=1e-14,rtol=0)
    # One grid over all five outer splits exercises held-out score collection;
    # the test does not perform a new four-grid model-selection experiment.
    training.validate_grid(events,protocol["test_pairs"],(10,6),args.output)
    metrics=training.collect_validation(args.output,[[10,6]])
    assert np.isfinite(metrics.nll).all()
    pipeline.refit_production(events,protocol,args.output)
    expected=np.load(inputs/"reference/production/xt.npy")
    np.testing.assert_allclose(np.load(args.output/"production/xt.npy"),expected,atol=1e-14,rtol=0)
    (args.output/"verification.json").write_text(json.dumps(dict(status="passed",bandwidths=bandwidths,
        six_match_max_difference=float(abs(fitted["xt"]-reference).max()),production_checked=True),indent=2)+"\n")


if __name__=="__main__":
    main()
