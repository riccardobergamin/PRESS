"""Selected real possessions: export checkpoint values and plot actual frames."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from PRESS import PRESS
from PRESS.visualization import plot_frame
from review.io import new_folder, read_table


def export_cases(states, possessions, case_file, output, tracking_root=None):
    """Fixed case keys are reviewed examples, not an automatic pattern ranking.

    Optional plotting uses the package's generic observed-frame plot. Scores
    in panel titles are held-out; production models never replace these h.
    A next-control frame comes from verified metadata, never an invented frame.
    """
    folder = new_folder(Path(output) / "cases")
    cases = json.loads(Path(case_file).read_text())
    selected_rows, summaries = [], []
    for case in cases:
        match, possession = str(case["match_id"]), str(case["possession_id"])
        rows = states.loc[states.match_id.eq(match) & states.possession_id.eq(possession)].sort_values("frame_start")
        parent = possessions.loc[possessions.match_id.eq(match) & possessions.possession_id.eq(possession)]
        if rows.empty or len(parent) != 1:
            raise ValueError(f"Case not in the band cohort: {match}/{possession}")
        parent = parent.iloc[0]
        if not bool(parent[case["pattern_flag"]]):
            raise ValueError(f"Case no longer meets its pattern: {case['case']}")
        selected_rows.append(rows.assign(case=case["case"]))
        summaries.append(dict(case=case["case"], **parent.to_dict()))
        if tracking_root is not None:
            tracking = read_table(Path(tracking_root) / "matches" / match / f"{match}_tracking.parquet")
            press = PRESS(tracking_data=tracking)
            panels = [(int(row.frame_start), f"h = {row.h:.1%}") for row in rows.itertuples()]
            next_frame = parent.get("next_control_frame")
            if pd.notna(next_frame):
                panels.append((int(next_frame), "Next verified individual control"))
            for index, (frame, title) in enumerate(panels):
                plot_frame(press, frame, title=f"Case {case['case']} · {parent.player_name} · {title}", show=False)
                plt.gcf().savefig(folder / f"case_{case['case']}_{index:02d}.png", dpi=180, bbox_inches="tight")
                plt.close(plt.gcf())
    pd.concat(selected_rows, ignore_index=True).to_csv(folder / "checkpoints.csv", index=False)
    pd.DataFrame(summaries).to_csv(folder / "possessions.csv", index=False)
