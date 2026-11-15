"""Render final BCOS on the documented real frame; no estimation or retraining."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PRESS import PRESS
from PRESS.visualization import plot_bcos


def main(match_dir, output_root):
    output_root = Path(output_root)
    press = PRESS.from_skillcorner(match_dir)
    frame = 3800
    result = press.compute_bcos(frame)
    fig, ax = plot_bcos(press, frame, show=False, figsize=(11, 7),
        title=f"Brisbane FC–Perth Glory · F. Bérenguer · 06:19\nBCOS = {result['BCOS area']:.2f} m²")
    fig.savefig(output_root/'figures/vignette_10_bcos_transport.png', dpi=180)
    plt.close(fig)
    summary = dict(match_id=press.match_info.match_id, frame_id=frame,
        formulation='transport', bcos_area_m2=result['BCOS area'],
        max_probability_mass_error=result['max_mass_error'],
        n_targets=len(result['grid']), params=result['params'])
    # Infinity is the terminal-event limit; make it explicit in public JSON.
    summary['params']['time'] = 'infinity (terminal probabilities)'
    (output_root/'data/examples/bcos/final_example_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('match_dir', type=Path)
    p.add_argument('--output-root',type=Path,default=Path('.'))
    a=p.parse_args();main(a.match_dir,a.output_root)
