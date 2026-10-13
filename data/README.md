# Data setup

Raw tracking and event files are not stored here. Obtain them from [SkillCorner opendata](https://github.com/SkillCorner/opendata) and follow its attribution terms [@skillcorner-opendata]. The source repository contains a MIT license file for software; no separate data redistribution grant was verified, so keep the raw files and real-match derived outputs outside this repository.

Set `PRESS_DATA_DIR` to the local `data` directory when running data-dependent preprocessing. The original preprocessing and possession-segmentation scripts live in the sibling research repository at `skillcorner-analysis/scripts/analysis/preprocess_matches.py` and `skillcorner-analysis/scripts/analysis/ball_drive_segmentation.py`; they currently use machine-specific paths and are documented as source references rather than moved or modified here.

The committed vignette inputs are synthetic and clearly labeled. They are sufficient to rebuild all committed figures and notebook outputs.
