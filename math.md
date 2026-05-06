Here's the complete mathematical recipe for the best model — **FAE features with KNN (K=4)** — written as a numbered procedure you can implement directly.

## Notation reference

- $N = 20$: number of frames per sequence
- $J = 12$: number of joints in the pose
- $J_m$: number of moving joints you choose (recommend $J_m = 4$: both wrists and both elbows)
- $P_t^{(j)} = (x_t^{(j)}, y_t^{(j)})$: 2D pixel coordinate of joint $j$ in frame $t$
- $i$: index over training examples
- $C$: number of classes (6 for BoxingVI, 14 for the original paper)

Joint indexing convention I'll use throughout (matching the paper's joint list):

| Index | Joint |
|---|---|
| 0 | Left shoulder |
| 1 | Right shoulder |
| 2 | Left elbow |
| 3 | Right elbow |
| 4 | Left wrist |
| 5 | Right wrist |
| 6 | Left hip |
| 7 | Right hip |
| 8 | Left knee |
| 9 | Right knee |
| 10 | Left ankle |
| 11 | Right ankle |

The moving joints are indices $\{2, 3, 4, 5\}$. The stable joint sources are indices $\{0, 1, 6, 7\}$.

---

## Step 1: Construct the two stable points per frame

For each frame $t \in \{0, 1, \ldots, 19\}$:

**Neck** (midpoint of shoulders):
$$P_t^{(\text{neck})} = \frac{P_t^{(0)} + P_t^{(1)}}{2}$$

**Pelvis** (midpoint of hips):
$$P_t^{(\text{pelvis})} = \frac{P_t^{(6)} + P_t^{(7)}}{2}$$

You now have two extra "virtual joints" per frame.

---

## Step 2: Compute the sequence-level reference point

Average both stable points across all 20 frames into a single 2D point:

$$\bar{r} = \frac{1}{2N} \sum_{t=0}^{N-1} \left( P_t^{(\text{neck})} + P_t^{(\text{pelvis})} \right)$$

This produces one fixed $(x, y)$ point per pose sequence. It's roughly the boxer's torso center over the whole window.

---

## Step 3: Center every joint coordinate

For every frame $t$ and every joint $j$ (including the derived neck and pelvis), subtract $\bar{r}$:

$$\tilde{P}_t^{(j)} = P_t^{(j)} - \bar{r}$$

From here on, all coordinates refer to the centered version. I'll drop the tilde for brevity.

---

## Step 4: Compute angular features per frame

For each frame $t$ and each moving joint $\ddot{u} \in \{2, 3, 4, 5\}$:

**Step 4a:** Build the two vectors from the moving joint to each stable point:

$$b_{\ddot{u}, \text{neck}, t} = P_t^{(\text{neck})} - P_t^{(\ddot{u})}$$
$$b_{\ddot{u}, \text{pelvis}, t} = P_t^{(\text{pelvis})} - P_t^{(\ddot{u})}$$

Each is a 2D vector.

**Step 4b:** Compute the cosine-based angular encoding:

$$\theta_t^{(\ddot{u})} = 1 - \frac{b_{\ddot{u}, \text{neck}, t} \cdot b_{\ddot{u}, \text{pelvis}, t}}{\|b_{\ddot{u}, \text{neck}, t}\| \cdot \|b_{\ddot{u}, \text{pelvis}, t}\|}$$

The dot product expanded:
$$b_{\ddot{u}, \text{neck}, t} \cdot b_{\ddot{u}, \text{pelvis}, t} = b_{\ddot{u}, \text{neck}, t}^x \cdot b_{\ddot{u}, \text{pelvis}, t}^x + b_{\ddot{u}, \text{neck}, t}^y \cdot b_{\ddot{u}, \text{pelvis}, t}^y$$

The norms:
$$\|b_{\ddot{u}, \text{neck}, t}\| = \sqrt{(b^x)^2 + (b^y)^2}$$

The result $\theta_t^{(\ddot{u})}$ is a single scalar in $[0, 2]$.

**Step 4c:** Stack the four moving-joint angles into a vector for this frame:

$$\theta_t = \begin{bmatrix} \theta_t^{(2)} \\ \theta_t^{(3)} \\ \theta_t^{(4)} \\ \theta_t^{(5)} \end{bmatrix} \in \mathbb{R}^4$$

After doing this for all 20 frames, you have a `(20, 4)` array of angles for the sequence.

---

## Step 5: Compute angular velocity (first central difference)

For frames $t = 1$ through $t = 18$ (you can't compute velocity at frames 0 or 19):

$$\theta_{v, t} = \theta_{t+1} - \theta_{t-1}$$

This is element-wise subtraction, producing another vector in $\mathbb{R}^4$.

After this step, you have an `(18, 4)` array of angular velocities.

---

## Step 6: Compute angular acceleration (second central difference)

For frames $t = 2$ through $t = 17$ (you can't compute acceleration at frames 0, 1, 18, or 19):

$$\theta_{a, t} = \theta_{t+2} + \theta_{t-2} - 2\theta_t$$

This is element-wise. Result: a `(16, 4)` array of angular accelerations.

---

## Step 7: Align all three quantities to the same frames

You have:
- $\theta_t$ defined for $t \in \{0, \ldots, 19\}$ (20 frames)
- $\theta_{v,t}$ defined for $t \in \{1, \ldots, 18\}$ (18 frames)
- $\theta_{a,t}$ defined for $t \in \{2, \ldots, 17\}$ (16 frames)

To stack them, you need the common range — frames 2 through 17, which is 16 frames.

So you keep:
- $\theta_t$ for $t \in \{2, \ldots, 17\}$: shape `(16, 4)`
- $\theta_{v, t}$ for $t \in \{2, \ldots, 17\}$: shape `(16, 4)`
- $\theta_{a, t}$ for $t \in \{2, \ldots, 17\}$: shape `(16, 4)`

---

## Step 8: Build the per-frame feature vector

For each valid frame $t \in \{2, \ldots, 17\}$, concatenate the three quantities:

$$f_t = \begin{bmatrix} \theta_t \\ \theta_{v, t} \\ \theta_{a, t} \end{bmatrix} \in \mathbb{R}^{12}$$

That's 4 angles + 4 velocities + 4 accelerations = 12 values per frame.

---

## Step 9: Concatenate across all valid frames

Build the final FAE feature vector for one pose sequence:

$$\mathbf{f}_{\text{FAE}} = \begin{bmatrix} f_2 \\ f_3 \\ f_4 \\ \vdots \\ f_{17} \end{bmatrix} \in \mathbb{R}^{192}$$

That's 16 frames × 12 values per frame = 192 features.

This is your one feature vector per punch.

---

## Step 10: Build the dataset matrix

Apply Steps 1–9 to every punch sequence in your training set. Stack into a matrix:

$$\mathbf{X} \in \mathbb{R}^{n \times 192}$$

where $n$ is the total number of training punches, and each row is one punch's feature vector.

Stack the corresponding labels:

$$\mathbf{y} \in \{0, 1, \ldots, C-1\}^n$$

For BoxingVI, $C = 6$.

---

## Step 11: KNN classifier — the math at training time

KNN doesn't actually train. It just stores the dataset:

$$\text{Memory} := (\mathbf{X}, \mathbf{y})$$

That's the entire "training" step.

---

## Step 12: KNN classifier — the math at inference time

Given a new punch's feature vector $\mathbf{f}^* \in \mathbb{R}^{192}$:

**Step 12a:** Compute the Euclidean distance from $\mathbf{f}^*$ to every training point $\mathbf{f}_i$ for $i = 1, \ldots, n$:

$$d_i = \|\mathbf{f}^* - \mathbf{f}_i\|_2 = \sqrt{\sum_{k=1}^{192} (f^*_k - f_{i,k})^2}$$

**Step 12b:** Find the $K = 4$ smallest distances. Let $\mathcal{N}(\mathbf{f}^*)$ be the set of indices of the 4 nearest neighbors.

**Step 12c:** Compute distance-weighted votes for each class $c \in \{0, \ldots, C-1\}$:

$$\text{score}(c) = \sum_{i \in \mathcal{N}(\mathbf{f}^*)} \mathbb{1}[y_i = c] \cdot \frac{1}{d_i}$$

where $\mathbb{1}[\cdot]$ is the indicator function (1 if true, 0 otherwise).

**Step 12d:** Predict the class with the highest weighted score:

$$\hat{y}^* = \arg\max_c \; \text{score}(c)$$

That's the prediction.

---

## Step 13: Evaluation via 10-fold stratified cross-validation

Split your dataset into 10 folds while preserving class proportions. For each fold $k \in \{0, \ldots, 9\}$:

**Step 13a:** Use folds $\{0, \ldots, 9\} \setminus \{k\}$ as training data (90% of the dataset) and fold $k$ as test data (10%).

**Step 13b:** "Train" KNN on the training data (just store it).

**Step 13c:** For every test example, run Steps 12a–12d to get a prediction.

**Step 13d:** Compute metrics on the test fold. Let $TP_c$, $FP_c$, $FN_c$ be true positives, false positives, false negatives for class $c$:

$$\text{Accuracy}_k = \frac{\sum_c TP_c}{|\text{fold}_k|}$$

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$F_{1,c} = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

Macro-averaged versions:

$$\text{Precision}_{\text{macro}, k} = \frac{1}{C} \sum_{c=0}^{C-1} \text{Precision}_c$$

(Same formula structure for recall and F1.)

**Step 13e:** After all 10 folds, compute the final reported metrics:

$$\mu_{\text{accuracy}} = \frac{1}{10} \sum_{k=0}^{9} \text{Accuracy}_k$$

$$\sigma_{\text{accuracy}} = \sqrt{\frac{1}{10} \sum_{k=0}^{9} (\text{Accuracy}_k - \mu_{\text{accuracy}})^2}$$

Report as $\mu \pm \sigma$.

---

## Step 14: Train the final deployment model

Once you've validated performance via cross-validation, train the final model on **all** your data — no held-out fold:

$$\text{FinalModel} := (\mathbf{X}_{\text{all}}, \mathbf{y}_{\text{all}})$$

Save this for deployment.

---

## Step 15: Inference on a new boxing video

For a new clip:

1. Run pose estimation → get pose sequence
2. Detect punch instance (or use pre-trimmed clip's middle frame)
3. Extract 20-frame window around the punch peak
4. Run Steps 1–9 to compute $\mathbf{f}^*_{\text{FAE}} \in \mathbb{R}^{192}$
5. Run Steps 12a–12d using the FinalModel to get $\hat{y}^*$
6. Output the predicted punch class

---

## Quick sanity-check on dimensions

For a $J_m = 4$ moving joints, $N = 20$ frames pipeline:

| Quantity | Shape | Element type |
|---|---|---|
| Raw pose sequence | $(20, 12, 2)$ | pixel coordinates |
| Centered pose sequence | $(20, 12, 2)$ | centered pixel coordinates |
| Per-frame angles $\theta_t$ | $(20, 4)$ | scalars in $[0, 2]$ |
| Velocities $\theta_{v,t}$ | $(18, 4)$ | scalars |
| Accelerations $\theta_{a,t}$ | $(16, 4)$ | scalars |
| Per-frame features $f_t$ | $(16, 12)$ | scalars |
| Final FAE vector $\mathbf{f}_{\text{FAE}}$ | $(192,)$ | scalars |

For 6,000 training examples, your dataset matrix $\mathbf{X}$ is $(6000, 192)$ — a very manageable size.

---

## Key implementation gotchas

A few things that bite people when implementing this:

**Numerical stability in Step 4b.** If a moving joint coincides exactly with the neck or pelvis, you'll divide by zero. In practice this almost never happens (wrist isn't at the neck), but add a small epsilon to the denominator just in case:

$$\theta_t^{(\ddot{u})} = 1 - \frac{b_{\ddot{u}, \text{neck}, t} \cdot b_{\ddot{u}, \text{pelvis}, t}}{\|b_{\ddot{u}, \text{neck}, t}\| \cdot \|b_{\ddot{u}, \text{pelvis}, t}\| + \epsilon}$$

with $\epsilon = 10^{-8}$.

**Feature scaling.** The angles $\theta$ are in $[0, 2]$, but velocities and accelerations have different ranges. KNN with Euclidean distance is sensitive to feature scale. Consider standardizing the FAE vector so each dimension has zero mean and unit variance:

$$f_k \leftarrow \frac{f_k - \mu_k}{\sigma_k}$$

where $\mu_k, \sigma_k$ are the mean and standard deviation of feature $k$ across the training set. The paper doesn't explicitly mention this, but it usually helps KNN.

**Stratified shuffling.** When splitting into 10 folds, use stratified shuffling to ensure each fold has roughly the same class distribution. With class imbalance (some classes have 46 samples, others 105), random splits can produce folds where rare classes are missing entirely.

**Same random seed across feature comparisons.** When you compare UAE vs 2DMDD vs FAE, use the same fold splits for fair comparison. Set `random_state=42` (or whatever) consistently.

---

## The whole math in one diagram

```
Pose sequence (20×12×2)
    ↓
Center on torso reference (Steps 1-3)
    ↓
Angles per frame (Step 4) → (20, 4)
    ↓
Velocities (Step 5)        → (18, 4)
Accelerations (Step 6)     → (16, 4)
    ↓
Align to common frames (Step 7) → all (16, 4)
    ↓
Stack per frame (Step 8) → (16, 12)
    ↓
Flatten (Step 9) → (192,)
    ↓
KNN with K=4, distance-weighted (Steps 11-12)
    ↓
Predicted class
```

That's the complete mathematical pipeline for the paper's best model. Want me to translate this into clean Python code now, with NumPy operations for each step?