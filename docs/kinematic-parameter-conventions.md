# Kinematic parameter conventions

The constants are motivated first by published movement categories and
arrival-model assumptions, then checked against the tracking distribution.
The descriptive rules were engineered **after inspecting the data and the
existing model settings**. They provide a transparent consistency check,
not an independent or prospectively specified estimation of physical capacity.

The final descriptive convention agreed on 26 October 2026 uses finite
speed **v > 2 m/s and Q95**. For acceleration magnitude, report **multiple
post-hoc rules**: thresholds **a > 3 or a > 4 m/s²** and **Q95, Q98, Q99**,
identifying whether the unit is a frame or a maximum per event.
No single acceleration quantile rule is claimed to be uniquely appropriate.
These thresholds select moving/high-acceleration observations
for the audit. They do not filter training examples, clip tracking values,
or change the fitted models. Acceleration magnitude includes changes in
direction and deceleration; it is not exclusively positive tangential acceleration.

For the carrier-versus-defender speed comparison within engagements, apply
v >2 simultaneously to BOTH players at the same frame. The paired Q95 values
are6.777970 and7.173692m/s, a5.5163% difference, on61281engagement-frame pairs.
An overlapping engagement can repeat a carrier-frame. This is a descriptive
check of the literature-motivated reduction, not a direct cap estimator.
The general off-ball pass-risk speed sample retains its individual v >2 rule.
**No speed threshold is applied to acceleration audits**: accelerations
starting from rest or low speed are retained.

## Measurements and sampling units

Player velocities are the first Savitzky–Golay derivatives of tracking
coordinates at 10 Hz, with a centred seven-frame, degree-two window.
Accelerations are derivatives of those velocity components using the same
window. Magnitudes are Euclidean norms. Differentiation is performed within
contiguous finite runs without crossing gaps or periods. This is offline
smoothing, with future support, rather than a causal velocity estimator.

Quantiles must identify their population and sampling unit. A player-frame
quantile weights time spent in a state; a quantile of maximum acceleration
per engagement weights engagement peaks. The two are reported separately.
Q95 describes upper running speeds. Alternative upper acceleration quantiles
make the sensitivity visible. These are post-hoc descriptive rules, rather
than a claim that the precise quantile uniquely determines the model setting.

The earlier Q95/Q99 checks remain below as sensitivity checks.

| Population and unit | Filter | Q95 | Q99 | Model setting |
|---|---|---:|---:|---|
| Defensive engager event-frames, matched OBE sample | v > 2 m/s | 7.050 | 8.066 | Defender speed 7 m/s in ball drive and BCOS |
| Current off-ball player-frames | v > 2 m/s | 5.745 | 7.100 | Pass-risk speed 5.5 m/s, previously tested against 5.0 |
| Current off-ball player-frames | a > 3 m/s² | 5.407 | 7.013 | Linear acceleration 7 m/s² |
| Current unique carrier frames within matched OBE | v > 2 m/s | 6.688 | 7.815 | Approved BCOS carrier speed 6.6 m/s for the new feature pipeline |
| Defensive engager acceleration peak per matched OBE, engagement sample | peak a > 3 m/s² | 6.000 | 6.966 | Additional engagement-specific support for acceleration 7 m/s² |
| Carrier acceleration peak per matched OBE, current audit | peak a > 3 m/s² | 5.449 | 6.682 | Carrier acceleration 7 m/s² is an approximate upper scale |

The off-ball definition is player ID different from the annotated carrier;
it includes goalkeepers and frames without an annotated carrier, where all
tracked players qualify. Restricting to known-carrier frames gives acceleration
Q99 = 6.487 m/s². The engagement defender audit uses the engagement-specific
tracking sample; off-ball and carrier audits use the general tracking sample. These are
different populations, not interchangeable estimates of one distribution.
Defender event-frames may repeat overlapping engagements; carrier frames
in the current carrier audit are unique by match and frame.

## Theory and literature

[Bradley et al. (2009), *High-intensity running in English FA Premier League
soccer matches*](https://doi.org/10.1080/02640410802512775) classify jogging
from 7.2 km/h (2 m/s), high-speed running from 19.8 km/h (5.5 m/s), and
sprinting above 25.1 km/h (approximately 7 m/s). These are operational
categories, not universal physiological boundaries. Thus 5–5.5 m/s is a
reference near entry into high-speed running, while 7 m/s is a pre-sprint
reference. The 2 m/s filter removes stationary/walking observations;
speeds below it are not all stationary. Jogging and high-speed running
are separate categories, with an intermediate running category.

[Spearman (2018), *Beyond Expected Goals*, §3.1.1](https://static.hudl.com/craft/downloads/SSAC2018_Beyond_Expected_Goals.pdf)
models acceleration toward a destination up to a speed limit, using
5 m/s and 7 m/s². The paper motivates these as effective match movement
capabilities and uses common constants for simplicity. It does **not**
prescribe our thresholds or acceleration quantile rules.

Accordingly, the project uses 5.5 m/s for pass reachability and 7 m/s for
defensive engagement, supported by the corresponding task populations.
For acceleration, 7 m/s² is the maximal acceleration parameter in the
kinematic routine, inherited from the Spearman reference and consistent
with upper-tail tracking accelerations. Filtering a > 3 or >4 selects strong
accelerations for the descriptive checks; upper quantiles reduce reliance
on isolated maxima without defining a unique parameter estimator.
This does not prove a universal physiological maximum or identify each
player's capacity. Acceleration magnitude and a sprint test's propulsive
acceleration are also different measurements.

The literature motivates the scale and interpretation; the post-hoc rules
recover values of similar magnitude in our tracking. The exact percentile
is secondary to this consistency. The constants are rounded, task-specific
approximations; in particular, 5.5 and 6.6 do not equal the updated Q95 exactly.

### Running with the ball and the BCOS carrier cap

[Benhassen et al. (2026), *Effects of Dribbling Constraints on Sprint
Acceleration Performance and the Force-Velocity Profile according to Playing
Positions in Professional Soccer Players*](https://doi.org/10.5114/jhk/204820)
compare 30-m sprint tests with and without the ball in 52 professional players.
Their discussion reports a roughly **6% decrease in theoretical maximal
velocity V0** while dribbling. V0 is inferred from a force–velocity profile,
not an observed match-speed percentile or the BCOS cap. Nevertheless,
7 × 0.94 = **6.58 m/s** provides an indicative scale comparison: the chosen
6.6 m/s is about **5.7% below 7 m/s**, consistent in magnitude. This arithmetic
is a plausibility check, not a fitted transfer rule between the experiments.

[Carling (2010), *Analysis of physical activity profiles when running with
the ball in a professional soccer team*](https://doi.org/10.1080/02640410903473851)
reports match-play peak speeds with the ball of approximately 25 km/h
(about 6.9 m/s) across 30 French Ligue 1 matches. This supports the broad
scale but does not estimate a paired percentage reduction without the ball.

The carrier speed setting is **7 × (1−0.06) = 6.58 m/s, rounded to 6.6**,
using the literature reduction as an approximate modeling rule. In the
simultaneously-moving engagement sample, the carrier and defender Q95 values
are6.778 and7.174m/s (5.52% lower for the carrier), giving a descriptive
consistency check. The earlier separately-filtered unique carrier-frame
Q95=6.688m/s remains a sensitivity result. These checks support the broad
scale of **6.6 m/s as a uniform effective carrier speed limit in BCOS**;
the cap is not set equal to a tracking quantile.
It is not an absolute ceiling on real dribbling speed. No claim is made
that 7 and 6.6 were obtained from paired sprint tests on our players;
pass-risk speed 5.5 concerns a different task and is not the no-ball baseline
for this percentage comparison.

### Evidence for carrier acceleration

Spearman's 7 m/s² is a no-ball arrival-model reference, not a measured
dribbling limit. Extending this reference to the BCOS carrier is an explicit
modeling assumption checked against several post-hoc handcrafted rules.
For example, acceleration magnitude peaks within matched carrier engagements
give Q99 = **6.681980 m/s²** after >3, and Q98 = **6.953477 m/s²** after >4.
The paired defender Q98 after >4 is **6.972449 m/s²**. These peak quantiles
are distinct from frame quantiles and do not establish a sustainable rate.

Retain carrier a_linear = 7 as a rounded upper scale supported by this audit
and the reference routine. Dribbling can constrain expressed acceleration,
but the evidence does not establish a universal proportional reduction of
this constant parameter. A lower speed cap does not mathematically require
a lower acceleration cap. No force-profile parameter or percentage change
in that parameter is used to justify the acceleration setting. Benhassen
is cited above solely for the speed comparison. The separately configurable
turning scale remains an approximation, not an independently calibrated
carrier capability.

### Additional evidence on acceleration while dribbling

[Rovan (2015), doctoral thesis, §§6.2.2 and 7.2](https://www.fsp.uni-lj.si/COBISS/Dr/Doktorat22990269RovanKlemen.pdf)
measures eight players on grass with 20-Hz GNSS, separating longitudinal
and transverse acceleration by step during dribbling turns. The dependence
on entry speed, turn and step supports task-specific kinematics, but does
not establish a constant sustainable 7 m/s² or a universal reduction factor.
This is a small controlled study, not professional match validation.

[Piras et al. (2017)](https://doi.org/10.1055/s-0043-118340) find a higher
energy cost with the ball during constant-speed runs. This supports an
additional technical/physiological demand, not a numerical acceleration cap.

The subsequent same-cohort audit compares the acceleration norm of carrier
and defender in all **8,234** current-v5 engagements on identical finite frame
support. With a >3, carrier/defender Q99 is **6.151752/6.558217** for event-frames
(carrier 6.20% lower) and **6.681980/6.969092** for event peaks (4.12% lower).
With a >4, the corresponding frame contrast is 1.63% lower and event-peak
contrast 0.84% higher. A universal percentage reduction is not established.
No outcome selection, F0 transfer or alternative acceleration decomposition
is used in this comparison. These are descriptive contrasts, not causal
paired sprint effects or validation of a sustainable acceleration.

The speed threshold2m/s excludes the quasi-static/walking band in Bradley's
classification. For acceleration, [Akenhead et al.2013](https://doi.org/10.1016/j.jsams.2012.12.005)
use >3m/s² for high acceleration; [Sonderegger et al.2016](https://doi.org/10.1371/journal.pone.0166534)
discuss both >3 and >4 absolute cutoffs. They are operational selection
thresholds; our upper-quantile rule is a post-hoc engineering choice.


Linear and turning acceleration remain separate user parameters. Setting
both to 7 in the benchmark is a parsimonious modeling convention: observed
acceleration magnitudes do not independently validate the fixed-position
turning approximation or estimate its effective turning scale.

The initial measured velocity is preserved when it exceeds the model speed
limit; no further acceleration is added in that case. The physical ball-drive
carrier follows its measured constant velocity and has no carrier speed cap.
BCOS instead uses the uniform carrier cap above. Learned arrival dispersion
s remains specific to the corresponding ball-drive training fold.

These constants are checked descriptively on the ten matches, without outcome
selection; they are not claimed to have been estimated separately in each fold.
The approved carrier-cap revision from 6.5 to 6.6 is applied in the deferred
research pipeline for new BCOS feature generation. It requires regenerating
those features before downstream training. Existing deployed model weights,
reported benchmarks and the legacy package feature defaults are unchanged;
deployment follows the new training and validation.

## Pass-risk receivers during flight

The descriptive receiver audit uses the5742successful passes in the frozen
6706-pass benchmark, from release inclusive to matched reception exclusive.
No missing/nonfinite kinematic windows occur. Receiver speed>2 gives frame
Q95=6.910999m/s; this population does not recover5.5 as Q95. Generic off-ball
and receiver-only speed distributions therefore have different interpretations.
Acceleration maxima per pass give Q99=6.724812m/s² after>3 and Q98=7.067928m/s²
after>4. All thresholds and quantiles are saved alongside frame results.
Two same-passer/receiver annotations are reported, retained in the overall
audit and separately excluded only in a descriptive sensitivity. Training
and fitted parameters are unchanged. Their exclusion changes the speed Q95
to6.912884m/s and leaves the acceleration peak Q98 after>4 unchanged.
Close descriptive quantiles do not prove that parameter changes have small
prediction effects; that question requires a separate inference comparison.

## Audit sources

The published summaries are [engagement quantiles](../data/benchmarks/ball_drive/engagement_quantiles.csv) and the [defender-speed check](../data/examples/ball_drive/defender_speed_check.csv).

The earlier speed > 1 m/s analyses remain sensitivity checks;
they must not be relabelled as speed > 2 m/s results.
