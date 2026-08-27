# Proposal: Constraint Geometric Attention for Wireframe Generation

Date: 2026-07-09
Author: Codex, for Claude/Fable handoff
Status: Proposal for the next experiment after the current wireflow baseline

## 1. Summary

The final target is:

```text
constraint points + nearby assembly context
  -> predicted part wireframe
  -> shell mesh
  -> eventually B-Rep
```

The current wireflow experiment asks a model to move random points inside a bbox until they match a typed wireframe. That is useful as a diagnostic, but the next structural step should be:

```text
constraint points
  -> constraint relation graph
  -> geometric wireframe graph
```

This proposal adds a first-stage **geometric attention model**. It predicts how geometrically close or related two constraint points should be on the eventual sheet-metal part, using only information available at generation time.

The first MVP intentionally avoids CATIA CAE stress/strain fields because they may require high manual/setup cost. Stress, strain, load-path, and topology-optimization-style signals should be future extensions. The first experiment should start with **geometric attention** only.

## 2. Minimal Input And Output

### Input

For each constraint point `i`:

```text
p_i: 3D coordinate in part or assembly frame
n_i: local surface normal at the constraint point
optional later:
  joint type, bolt/weld/clip/mount type
  hole/washer radius
  support/load tag
  nearby part occupancy or clearance field
  assembly-level part identity/context
```

For the first experiment, keep the model as small and clean as possible:

```text
node_i = [normalized xyz, normal xyz]
```

Normalize coordinates by part/condition bbox center and scale. Preserve the world-to-part transform metadata so later joints and generated geometry remain consistent.

### Output

A symmetric pairwise attention matrix:

```text
A_ij in [0, 1]
```

Interpretation:

```text
A_ij high  = constraint points i and j are geometrically close/related on the final sheet-metal surface
A_ij low   = points are likely separated by long surface distance, different regions, or weak direct geometric coupling
```

This is not the transformer's internal attention. It is an explicit predicted artifact: a **constraint relation graph** that can be evaluated, visualized, and passed into the wireframe generator.

## 3. Supervision Targets

The target score should be derived from completed parts where the true midsurface/wireframe is known.

Preferred target if practical:

```text
d_geo(i, j) = shortest path distance between constraint points on the midsurface mesh
score(i, j) = exp(-d_geo(i, j) / tau)
```

If direct midsurface geodesic distance is slow or noisy, use staged approximations:

1. **Wireframe graph distance**
   - Project each constraint point to the nearest typed wireframe entity.
   - Compute shortest path distance along the wireframe graph.
   - Good for a first implementation because the current wireframe identification is already strong.

2. **Patch/region relationship**
   - Same patch or same loop neighborhood: positive.
   - Across one bend line: medium.
   - Across many bends or disconnected feature regions: low.

3. **Euclidean + normal heuristic teacher**
   - Use as a baseline only, not as final evidence.
   - Example: nearby points with compatible normals get higher score.

The model should report both regression and ranking metrics. The real question is not exact score calibration at first; it is whether the model ranks geometrically relevant point pairs above irrelevant ones.

## 4. Why This Helps Wireframe Generation

Constraint points alone are underdetermined. A bbox and a few holes/mounts do not specify where loops, bend lines, and surface frames should go. The geometric attention matrix provides the missing relational prior:

```text
which constraints likely belong to the same sheet region
which constraints should be connected by material
which constraints are separated by long folded paths
which groups define local loops, ribs, or mounting regions
```

Then the wireframe generator can condition on:

```text
nodes: constraint points with normals
edges: predicted A_ij scores
global: bbox / scale / optional surrounding parts
```

This changes the problem from:

```text
generate a wireframe from independent points
```

to:

```text
generate a wireframe from a constraint relation graph
```

That is a much better fit for sheet metal because the final wireframe is itself a geometric graph/cell complex.

## 5. Recommended Model

Start with a pairwise geometric graph model:

```text
Input constraint nodes:
  xyz, normal

Pair features:
  delta xyz
  distance
  normal dot product
  projected distance along normals/tangent plane
  relative normal-angle features

Encoder:
  small Transformer or GNN over constraint nodes

Pair head:
  h_i, h_j, pair_features_ij -> score A_ij
```

Keep invariance in mind:

- Translation invariance: subtract bbox center.
- Scale robustness: divide by bbox diagonal or max extent.
- Rotation robustness: use relative pair features and normal dot products.
- Do not overfit to global vehicle coordinates unless assembly orientation is intentionally part of the condition.

Candidate losses:

```text
L_distance = SmoothL1(predicted_distance, target_geodesic_distance_normalized)
L_score    = BCEWithLogits(predicted_score, target_close_pair)
L_rank     = InfoNCE / pairwise ranking loss for top-k related neighbors
```

A practical first version can train only a score head with BCE/ranking labels:

```text
positive pairs = top-k nearest pairs by true geodesic or wireframe distance
negative pairs = far pairs sampled from the same part
```

## 6. Evaluation

Evaluate the attention model before integrating it into wireframe generation.

Metrics:

```text
top-k recall:
  For each constraint point, does predicted top-k include the true geodesic-nearest neighbors?

Spearman / Kendall rank correlation:
  Does predicted score order match true geometric distance order?

AUC / average precision:
  Can the model separate close-pair vs far-pair labels?

distance bucket accuracy:
  near / medium / far classification
```

Baselines:

```text
Euclidean distance only
Euclidean distance + normal dot
nearest-neighbor by bbox-normalized coordinate
oracle score from true geodesic/wireframe distance
```

The geometric attention model is useful only if it beats Euclidean+normal baselines on held-out parts. If it does not, feed the heuristic score directly to the wireframe generator and avoid adding a weak learned component.

## 7. Integration Into Wireframe Generation

After the attention model is validated, use `A_ij` in the wireframe generator in three ways.

### 7.1 Attention Bias

Pass `A_ij` as a bias into cross-attention between generated wireframe tokens and constraint tokens.

```text
attention_bias_ij = alpha * logit(A_ij)
```

This is the least invasive integration path for the current wireflow work.

### 7.2 Candidate Edge Prior

Use high-score pairs as candidate material paths or patch-local relationships.

```text
high A_ij -> likely same local sheet region or connected feature group
low A_ij  -> avoid direct generated connections unless other evidence supports them
```

This is especially useful when moving from point generation to geometric wireframe graph generation.

### 7.3 Patch Graph Prior

Cluster constraints by predicted relation score:

```text
constraint relation graph
  -> local groups
  -> likely patch/loop anchors
  -> outer/hole/bend/surface_frame generation
```

This supports the longer-term direction:

```text
constraint relation graph -> geometric wireframe -> patch graph -> shell mesh
```

## 8. Future Mechanical Attention

Do not start with full stress/strain fields. Keep them as planned extensions:

1. **CATIA CAE stress/strain attention**
   - Use FEA responses to derive pairwise compliance, load transfer, or stress propagation scores.
   - More physically meaningful, but higher setup cost.

2. **Topology optimization load-path attention**
   - Run coarse topology optimization or simplified shell/truss optimization.
   - Extract material/load-path skeletons as soft constraints.

3. **Hybrid score**
   - Combine:

```text
A_total = w_geo * A_geometric
        + w_mech * A_mechanical
        + w_context * A_surrounding_parts
```

The first geometric attention model should be designed so these additional channels can be added later without changing the downstream wireframe interface.

## 9. Proposed First Implementation Plan

### Phase 0: Dataset Builder

Create a dataset from assemblies with reliable joints and wireframes:

```text
input:
  constraint point xyz + local normal

target:
  pairwise geometric score from wireframe graph distance or midsurface geodesic distance
```

Start with `A0072600002_AllCATPart` and `A0072601285_AllCATPart`. Exclude assemblies without reliable annotations for the first training run.

### Phase 1: Baseline And Learned Attention

Implement:

```text
baseline_geometric_score.py
train_constraint_attention.py
evaluate_constraint_attention.py
```

Report:

```text
Euclidean baseline
Euclidean+normal baseline
learned score
oracle score
```

### Phase 2: Wireflow Conditioning

Use the learned score as an attention bias or extra pair feature in the current wireflow decoder. Compare:

```text
wireflow without A_ij
wireflow with heuristic A_ij
wireflow with learned A_ij
oracle A_ij upper bound
```

### Phase 3: Geometric Wireframe Graph Generator

Move away from unordered random point convergence and generate an explicit geometric graph:

```text
vertices
curves/edges
typed loops
patch incidence
```

The constraint attention graph becomes the natural conditioning structure for this generator.

## 10. Key Guardrails

- Do not call the score "mechanical" until it uses real mechanical supervision or simulation.
- Keep `A_ij` inspectable. Save it as JSON/CSV and visualize it as colored edges between constraint points.
- Compare against Euclidean+normal before claiming learning value.
- Treat exact geodesic regression as secondary; top-k relation recovery is the first success criterion.
- Avoid leakage: compute target distances only from the training target geometry, never from validation geometry at inference time.
- Preserve part-local/world transforms for all constraint coordinates and normals.

## 11. First Success Criterion

The first milestone is not improved wireframe generation. It is:

```text
Given only constraint point coordinates and local normals,
the model predicts geometrically related constraint pairs
better than Euclidean+normal baselines on held-out parts.
```

If this succeeds, the predicted relation graph should be integrated into wireframe generation. If it fails, the project still gains a useful conclusion: current constraint-point information is insufficient and needs surrounding part context, joint types, or mechanical signals.
