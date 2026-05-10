boxing_punch_classification_math
Classifying the 6 Boxing Punches from 3D Pose — A Pure-Math Playbook
A reference for rule-based / handcrafted-feature classification of {1: jab, 2: cross, 3: lead hook, 4: rear hook, 5: lead uppercut, 6: rear uppercut} from MotionBERT's 17-keypoint H36M output. No deep learning required.

0. The structure of the problem (the key insight)
The 6 classes form a 2×3 grid that you should exploit:

Straight	Hook (horizontal arc)	Uppercut (vertical arc)
Lead	1 Jab	3 Lead hook	5 Lead uppercut
Rear	2 Cross	4 Rear hook	6 Rear uppercut
This means you can decompose the 6-way problem into two independent binary/ternary decisions:

Which hand? (lead vs rear) — a clean binary call from "which wrist moved."
Which trajectory family? (straight / horizontal arc / vertical arc) — a 3-way call from the geometry of the wrist path and the elbow-angle profile.
Solving these two sub-problems separately is much easier than fighting all 6 jointly, and it matches how the biomechanics literature actually describes the punches: the elbow is the upper-body segment that contributes most to the cross (acting like a piston in flexion-extension), while the shoulder dominates during the hook and uppercut, which require circular trajectories.

1. Notation and preprocessing
1.1 Joints (MotionBERT / H36M-17 indices)
0  Pelvis (root)        8  Thorax
1  R hip                9  Neck/Nose
2  R knee              10  Head top
3  R ankle             11  L shoulder
4  L hip               12  L elbow
5  L knee              13  L wrist
6  L ankle             14  R shoulder
7  Spine (mid)         15  R elbow
                       16  R wrist
Let p_i(t) ∈ ℝ³ be the 3D position of joint i at frame t.

1.2 Body-centered, body-rotated frame (do this first)
Raw MotionBERT output is in camera space. Every feature below assumes you've moved into the boxer's own frame. Otherwise stance + camera angle will dominate the signal.

Origin: translate so root (pelvis) is at origin: p̃_i(t) = p_i(t) − p_0(t).

Orientation: build an orthonormal body frame from the shoulders and pelvis at a stable reference frame (e.g., the guard frame, see §3):

Right axis: x̂ = normalize(p_14 − p_11) (R shoulder − L shoulder)
Up axis: ẑ = normalize(p_8 − p_0) (thorax − pelvis), then re-orthogonalize against x̂
Forward axis: ŷ = ẑ × x̂ (points out of the chest)
Stack R = [x̂, ŷ, ẑ]ᵀ and rotate every joint: q_i(t) = R · p̃_i(t).

Now +y is "where the boxer is facing," +x is "to the boxer's right," +z is "up." This single trick fixes orthodox vs southpaw stance ambiguity, camera roll, and most of the view-invariance problem.

Why this matters: Skeleton-based action data has high inter-class similarity, but distinctive motion patterns become much more dominant when the 3D skeleton is viewed from the right direction. The body frame is that "right direction."

1.3 Scale normalization
Divide all positions by a stable body length (shoulder width or torso length):

L_torso = ||p_8 − p_0||             (thorax to pelvis)
q_i(t) ← q_i(t) / L_torso
Now distances are in "torso units" and the same rules work across body sizes.

1.4 Smoothing & derivatives
3D pose is noisy. Smooth before differentiating.

Savitzky–Golay filter (window 5–9, polyorder 2 or 3) is the standard for biomechanics — gives you smoothed position and analytic velocity/acceleration for free.
Velocity: v_i(t) = (q_i(t+1) − q_i(t−1)) / (2Δt)
Acceleration: a_i(t) = (q_i(t+1) − 2q_i(t) + q_i(t−1)) / Δt²
Speed: s_i(t) = ||v_i(t)||
2. The feature library
These are the building blocks that show up over and over in the skeleton-action-recognition literature: joint-joint distances (JJd), joint-joint orientations (JJo), joint-joint vectors (JJv), joint-line distances (JLd), and line-line angles (LLa).

2.1 Joint angle (3-point angle)
For three joints A–B–C with B as the vertex (e.g., shoulder–elbow–wrist for elbow flexion):

 
Numerically stable form: θ = atan2(||u × v||, u · v) where u = A−B, v = C−B.

Key angles for boxing:

Elbow flexion θ_elbow = ∠(shoulder, elbow, wrist) → ~180° = arm extended (jab/cross), ~90° = bent (hook/uppercut at impact)
Shoulder elevation θ_shoulder = ∠(elbow, shoulder, hip_same_side) → tells you if the arm is raised laterally (hook prep)
Trunk lean θ_trunk = ∠(neck, pelvis, world_up) → forward lean during cross, lateral lean during hook
Knee bend θ_knee = ∠(hip, knee, ankle) → uppercuts have a deep load phase
2.2 Signed plane angles (the secret weapon for hook vs uppercut)
A plain 3-point angle is unsigned and loses direction. You want signed angles in a chosen plane — this is what cleanly separates a horizontal arc (hook) from a vertical arc (uppercut).

Project the forearm vector f = wrist − elbow onto:

Horizontal plane (xy, removing z): f_h = (f_x, f_y, 0)
Sagittal plane (yz, removing x): f_s = (0, f_y, f_z)
Frontal plane (xz, removing y): f_f = (f_x, 0, f_z)
Then azimuth = atan2(f_x, f_y) gives the horizontal swing angle, and elevation = atan2(f_z, sqrt(f_x² + f_y²)) gives the vertical tilt.

A hook has high d(azimuth)/dt; an uppercut has high d(elevation)/dt.

2.3 Wrist trajectory descriptors
Treat the punching wrist's 3D path over the punch window [t_0, t_end] as a curve.

Path length:

Chord length (start to end displacement):

Straightness ratio (THE single most useful feature):

 
ρ ≈ 1 → straight punch (jab, cross)
ρ ≈ 0.5–0.8 → curved (hook, uppercut)
This is sometimes called the trajectory efficiency or directness index. It directly reflects the biomechanical difference: the cross is a straight trajectory, while the hook and uppercut require circular trajectories with shoulder-driven rotation and translation.

Curvature (Frenet form for discrete 3D points): at each frame,

 
Straight punch: max κ is small.
Hook/uppercut: max κ is large, occurring at the apex of the arc.
2.4 Plane of motion (PCA on the trajectory)
Run PCA on the wrist position points {q_wrist(t)} during the punch:

Eigenvalues λ_1 ≥ λ_2 ≥ λ_3. The eigenvector for λ_1 is the dominant direction; for λ_3, the trajectory's normal.
Planarity: (λ_1 + λ_2) / (λ_1 + λ_2 + λ_3) is near 1 for hooks/uppercuts (planar arcs), and the normal vector itself tells you which plane:
Normal ≈ ±ẑ (vertical) → motion is in the horizontal plane → hook
Normal ≈ ±x̂ (lateral) → motion is in the sagittal plane → uppercut
High λ_1 dominance with low planarity scores → straight punch
This is one of the cleanest single-shot tricks for the 3-way trajectory family decision.

2.5 Velocity and timing
Peak wrist speed s_max = max_t ||v_wrist(t)||. Reported ranges from camera-based 3D analysis: average velocities at impact range from 5.9 to 8.2 m/s with peaks of 6.6 to 12.5 m/s, reached 8 to 21 ms before contact.

Rough peak-speed ordering (useful as a tiebreaker, not a primary feature, since it varies with the boxer):

Jab < cross ≈ uppercut < hook (hooks tend to be fastest at the wrist due to swing)
Time-to-peak velocity and deceleration profile (the hand snaps back) help segment one punch from the next.

2.6 Body-segment velocity contributions
This is the gold standard from the biomechanics paper. The contribution of each body segment is found by projecting the velocity vector of the segment on the velocity vector of the wrist. For each segment s (pelvis, trunk, shoulder, elbow):

 
Then express each as a percentage of the wrist speed at impact. The pattern is diagnostic: at impact time, the elbow is the upper-body segment that contributes most to the cross, while it is the shoulder that contributes most during the hook and uppercut.

2.7 Hand-displacement geometry from the guard
Define the guard pose as the rest pose just before launch (see §3). Let Δ = q_wrist(t_impact) − q_wrist(t_guard).

Feature	Jab/Cross	Hook	Uppercut
Δ_y (forward)	large positive	medium	small
Δ_x (lateral)	small	large, sign = inside-to-target	small
Δ_z (up)	small	small	large positive
\|Δ\|	large	medium	medium
Δ_y / \|Δ\|	~1.0	~0.3	~0.4
Δ_z / \|Δ\|	small	small	large
These three ratios alone get you 80% of the way to a working classifier.

2.8 Statistical features over the punch window
Once you have time-series for the features above, compute summaries. Time-domain features (mean, standard deviation, max, min, interquartile range, entropy, skewness, kurtosis, mean absolute deviation) capture variations in punch intensity.

For each scalar signal f(t) in the punch window: [mean, std, min, max, range, skew, kurtosis, argmax (timing)].

3. Punch detection and segmentation (you need this before classification)
Two-state model: GUARD vs PUNCH.

3.1 Guard detection
A frame is "guard-like" if both wrists are:

close to the chin: ||q_wrist − q_neck|| small (in torso units, < ~0.5)
low speed: s_wrist < threshold
3.2 Punch onset / offset
Use a velocity threshold with hysteresis on the dominant wrist:

Onset t_0: first frame where s_wrist(t) crosses a high threshold (e.g., 3 m/s in real units, or 2× peak guard speed) going up.
Peak t_peak: argmax_t s_wrist(t) near the end of extension.
Impact / max-extension t_imp: local max of forward displacement OR local min of acceleration toward target (the "snap" deceleration).
Offset t_end: wrist returns to within ~0.3 torso units of the guard position.
3.3 Optional: zero-velocity segmentation
For continuous bag/shadow-boxing, find local minima of s_wrist(t) below a small threshold — these are the natural breakpoints between punches.

3.4 Robust onset
Use the jerk signal (3rd derivative) to find sudden launches; jerk peaks lead velocity peaks and are a clean trigger.

4. The 6-class classifier (a concrete decision logic)
Compute features over the window [t_0, t_imp] (extension phase). Classify in two stages.

Stage A — Which hand? (lead vs rear)
Trivial in principle, less so in practice (sometimes both hands move). Use:

score_R = max_t ||v_R_wrist(t)||  -  ||v_R_wrist(t_guard)||
score_L = max_t ||v_L_wrist(t)||  -  ||v_L_wrist(t_guard)||

active = argmax(score_R, score_L)
Then resolve to lead/rear from the stance — detect which foot is forward at the guard:

lead_side = "L" if q_L_ankle.y > q_R_ankle.y else "R" (orthodox = L lead)
is_lead_punch = (active == lead_side)
Stage B — Trajectory family (straight / hook / uppercut)
Compute on the active wrist over [t_0, t_imp]:

Symbol	Formula	What it captures
ρ	D / L	straightness
Δ_y/D	forward fraction of displacement	"is it a jab/cross?"
Δ_z/D	vertical fraction of displacement	"is it an uppercut?"
Δ_x/D	lateral fraction of displacement	"is it a hook?"
θ_elb,min	minimum elbow angle in the window	hook/uppercut bend much more
Δθ_elb	elbow angle change θ_max − θ_min	cross has the largest range (piston)
n̂	PCA-3 eigenvector of wrist trajectory	plane of motion
Decision rule (tunable, but a good starting point):

# 1. Straight punch?
if ρ > 0.85 and |Δ_y| / D > 0.7 and Δθ_elb > 70°:
    family = STRAIGHT
# 2. Uppercut?
elif Δ_z / D > 0.5 and θ_elb,min < 110° and |n̂ · x̂| > 0.7:
    family = UPPERCUT
# 3. Hook?
elif |Δ_x| / D > 0.5 and θ_elb,min < 110° and |n̂ · ẑ| > 0.7:
    family = HOOK
else:
    family = argmax of soft scores below
Soft scoring (better than hard cuts)
Define three "scores" and take the argmax — this is more robust than chained if/else:

score_straight = w1·ρ + w2·(Δ_y/D) + w3·(Δθ_elb / 180°)
score_uppercut = w4·(Δ_z/D) + w5·(1 − ρ) + w6·|n̂ · x̂|
score_hook     = w7·(|Δ_x|/D) + w8·(1 − ρ) + w9·|n̂ · ẑ|
Tune weights on a small labeled set. Even unit weights (w_i = 1) get you surprisingly far.

Final mapping
class = (is_lead_punch, family) →
    (lead, straight)    → 1 jab
    (rear, straight)    → 2 cross
    (lead, hook)        → 3 lead hook
    (rear, hook)        → 4 rear hook
    (lead, uppercut)    → 5 lead uppercut
    (rear, uppercut)    → 6 rear uppercut
