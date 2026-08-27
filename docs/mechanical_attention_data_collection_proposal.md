# Proposal: Mechanical Attention Data Collection for Constraint Graphs

Date: 2026-07-09
Author: Codex, for Claude/Fable handoff
Status: Proposal for future extension after geometric attention

## 1. Summary

The current direction is:

```text
constraint points + local normals
  -> geometric attention / constraint relation graph
  -> geometric wireframe
  -> shell mesh
  -> eventually B-Rep
```

This document proposes the next extension: **mechanical attention**.

The key idea is not to use stress/strain fields as direct inference-time inputs. At inference time the target part shape does not exist yet, so a true stress field cannot be computed. Instead, existing completed parts should be used to generate mechanical supervision:

```text
completed part + constraints + CAE
  -> per-load response fields
  -> pairwise mechanical attention matrix M_ij
  -> train a model to predict M_ij from constraint geometry/context
```

Then the generation pipeline becomes:

```text
constraint points + normals + surrounding context
  -> geometric attention G_ij
  -> predicted mechanical attention M_ij
  -> wireframe graph generation
```

Start with geometric attention first. Mechanical attention should be added after the geometric attention dataset/model is inspectable.

## 2. Why Pairwise Mechanical Attention

Raw stress fields are high-dimensional, mesh-dependent, boundary-condition-dependent, and expensive to generate. For wireframe generation, the most useful signal is usually not the full field itself, but the relation between functional points:

```text
If load enters at constraint i,
which other constraints and sheet regions participate in carrying it?
```

This can be stored as:

```text
M_ij = mechanical relatedness score between constraint i and constraint j
```

High `M_ij` means:

- force or displacement response transfers strongly between the two constraints
- the same material path or patch likely participates in both load cases
- the generated wireframe should probably preserve a strong sheet connection between their regions

Low `M_ij` means:

- the two constraints are mechanically weakly coupled
- direct material/feature connection is less likely unless required by geometry or packaging

## 3. First Data To Collect

For each completed part with reliable constraints and mesh:

```text
constraint i:
  xyz position
  local surface normal
  optional local tangent basis t1, t2
  joint type / washer radius / support type if available
```

For each constraint `i`, run unit load cases.

### Minimal Load Cases

Start with one load case per constraint:

```text
case i_N:
  apply unit force at constraint i along local normal n_i
```

This is the lowest-effort first dataset. It is enough to test whether mechanical attention adds any information beyond geometric attention.

### Preferred Load Cases

If CATIA CAE automation cost is acceptable, use three orthogonal directions:

```text
case i_N:
  unit force along normal n_i

case i_T1:
  unit force along tangent t1_i

case i_T2:
  unit force along tangent t2_i
```

Sheet metal often carries important in-plane/shear loads, so tangent loads will eventually be important.

### Optional Later Cases

Later, add:

```text
unit moment around n_i / t1_i / t2_i
pressure-like local patch load
realistic assembly load cases
thermal or vibration cases if relevant
```

Do not start here. First prove value with unit force cases.

## 4. Boundary Condition Protocol

Boundary conditions must be standardized. Otherwise the learned signal will mostly reflect inconsistent CAE setup choices.

Recommended first protocol:

```text
For load at constraint i:
  apply unit force at i
  all other constraints use the same standardized support model
```

Candidate support models:

1. **Fixed supports at all other constraints**
   - easiest to set up
   - stable
   - may be too stiff and overemphasize local peaks

2. **Spring supports at all other constraints**
   - preferred if practical
   - use the same translational/rotational stiffness for all parts
   - closer to assembly-like compliance

3. **Hybrid**
   - fixed normal direction, spring tangential directions
   - useful if rigid-body modes are troublesome

The support protocol must be written into the dataset manifest.

## 5. Fields To Export

For each load case, save both full fields and reduced summaries.

### Full Fields

Save these if CATIA/export allows it:

```text
node displacement vector u
element or node von Mises stress
principal stress values/directions
strain energy density
shell membrane/bending stress components if available
```

The most valuable full-field quantity is usually:

```text
strain energy density
```

Stress peaks can be noisy near point loads, holes, and mesh singularities. Strain energy density is a better load-path signal because it indicates where the structure is doing mechanical work.

### Reduced Per-Constraint Responses

For every load case at `i`, collect responses near every constraint `j`:

```text
u_j:
  displacement vector at/near constraint j

u_j_projected:
  displacement projected onto n_j, t1_j, t2_j

reaction_j:
  reaction force/moment at supported constraint j if available

energy_near_j:
  sum/mean/max strain energy density in a local neighborhood around j

stress_near_j:
  robust p95 von Mises stress near j, not raw max
```

Use local neighborhoods such as washer radius, fixed mm radius, or k-nearest shell elements. Record which definition is used.

## 6. Constructing Mechanical Attention Scores

Several pairwise scores should be generated and compared.

### 6.1 Compliance Response

```text
C_ij = norm(displacement at j under unit load at i)
```

or direction-aware:

```text
C_ij_ab = displacement of j along direction b under unit load at i along direction a
```

This approximates how much motion at one constraint influences another. It is close to a boundary compliance/influence matrix.

### 6.2 Reaction Transfer

```text
R_ij = norm(reaction at j under unit load at i)
```

This asks where the applied load is supported. It is often more intuitive for fixture/fastener relationships.

### 6.3 Strain Energy Near Target Constraint

```text
E_ij = strain energy near j under unit load at i
```

This is a local participation score.

### 6.4 Field Overlap / Shared Load Path

For each unit load case `i`, normalize the strain energy density field:

```text
e_i(x) = normalized strain energy density under load at i
```

Then:

```text
O_ij = cosine_similarity(e_i, e_j)
```

This is likely the best load-path relatedness score. It says whether two constraints activate similar material regions, even if they are not adjacent in Euclidean space.

### 6.5 Final Mechanical Attention

Do not choose one formula too early. Save multiple channels:

```text
M_ij = {
  compliance_response,
  reaction_transfer,
  target_neighborhood_energy,
  strain_energy_field_overlap
}
```

The wireframe generator can later use one channel, a learned combination, or a multi-channel edge feature.

## 7. Normalization

Mechanical values must be normalized for cross-part learning.

Recommended:

```text
coordinate scale: bbox diagonal or max extent
force scale: unit force, fixed for all parts
thickness: record explicitly; normalize stiffness/energy by thickness if needed
material: record E, nu, density; start with a single material if possible
energy fields: normalize each load case by total strain energy
reaction scores: divide by applied force magnitude
compliance scores: divide by part scale / force
```

Also save raw values. Normalization choices will change.

## 8. Dataset Schema

Suggested artifact per part:

```text
mechanical_attention/<assembly>/<part_id>/
  constraints.json
  cae_manifest.json
  load_cases.json
  pair_scores.npz
  fields/
    case_000_energy.vtk or npz
    case_000_displacement.vtk or npz
    ...
```

### constraints.json

```json
{
  "part_id": "...",
  "coordinate_frame": "part_local",
  "constraints": [
    {
      "id": 0,
      "xyz": [0.0, 0.0, 0.0],
      "normal": [0.0, 0.0, 1.0],
      "tangent_1": [1.0, 0.0, 0.0],
      "tangent_2": [0.0, 1.0, 0.0],
      "joint_type": "bolt_or_unknown",
      "radius_mm": null
    }
  ]
}
```

### cae_manifest.json

```json
{
  "solver": "CATIA_CAE",
  "element_type": "shell",
  "mesh_size_mm": 5.0,
  "thickness_mm": 1.2,
  "material": {"E": 210000.0, "nu": 0.3},
  "support_protocol": "other_constraints_fixed",
  "load_protocol": "unit_normal_force",
  "notes": ""
}
```

### pair_scores.npz

Store arrays:

```text
compliance[n_constraints, n_constraints]
reaction_transfer[n_constraints, n_constraints]
energy_near_target[n_constraints, n_constraints]
energy_field_overlap[n_constraints, n_constraints]
valid_mask[n_constraints, n_constraints]
```

If directional loads are included:

```text
compliance[n_constraints, n_constraints, n_load_dirs, n_response_dirs]
reaction_transfer[n_constraints, n_constraints, n_load_dirs]
```

## 9. First Experiment

Before generating large data, run a small calibration study.

Recommended size:

```text
10 to 20 parts
normal-load-only cases
same material/thickness if possible
fixed or spring support protocol
```

Questions to answer:

1. Are the matrices stable and interpretable?
2. Does `M_ij` correlate with geometric attention `G_ij`?
3. Where does `M_ij` disagree with `G_ij`?
4. Do disagreements explain meaningful structural relationships?
5. Can `M_ij` predict same-patch, same-load-path, or wireframe graph distance better than Euclidean distance?

If the answer is no, do not scale CAE yet.

## 10. Integration With Current Geometric Attention Work

Claude is currently working on the constraint relation graph. The clean integration path is:

```text
Stage A:
  geometric attention only

Stage B:
  mechanical attention teacher from CAE on a small subset

Stage C:
  train a mechanical attention predictor:
    input: constraint xyz + normal + optional context
    target: M_ij from CAE

Stage D:
  wireframe generator conditions on:
    G_ij
    predicted M_ij
    optional oracle M_ij upper bound for analysis only
```

Important: oracle `M_ij` from actual target geometry/CAE must never be used at inference or validation generation except as an upper-bound diagnostic.

## 11. User Work Items

The user is best positioned to prepare CAE-side data and validate engineering assumptions.

### Must Do First

1. Select 10 to 20 representative parts with reliable constraint points and wireframes.
2. Confirm whether CATIA CAE can automate:
   - shell mesh generation
   - assigning thickness/material
   - applying unit load at a selected constraint point
   - fixing or spring-supporting other constraint points
   - exporting displacement, stress, and strain energy density
3. Decide the first support protocol:
   - fixed other constraints, or
   - spring-supported other constraints
4. Decide the first load protocol:
   - normal-only, or
   - normal + two tangent directions
5. Run 1 to 2 pilot parts manually and export results.

### Nice To Have

1. Record CATIA screenshots/results for one easy-to-understand part.
2. Confirm whether strain energy density can be exported directly.
3. Confirm whether shell membrane/bending stress components are available.
4. Check how point loads are applied: exact node, washer patch, RBE/spider, or distributed local patch load.
5. Prefer distributed washer/patch loads over singular point loads if setup cost is reasonable.

### Avoid At First

1. Do not run hundreds of parts before validating the schema.
2. Do not mix inconsistent boundary conditions.
3. Do not use raw maximum stress as the main target.
4. Do not start with complicated real-world combined loads.
5. Do not make mechanical attention a required input for the first geometric attention generator.

## 12. First Success Criterion

The first milestone is:

```text
From completed parts and standardized CAE unit load cases,
construct inspectable pairwise mechanical attention matrices
that reveal plausible load-transfer or shared-load-path relationships
between constraint points.
```

The second milestone is:

```text
Train a predictor that estimates those matrices from constraint point coordinates,
normals, and optional context better than geometry-only baselines.
```

Only after these pass should mechanical attention be integrated into wireframe generation.

## 13. References For Direction

- Graph neural networks have been used to predict displacement, stress, and strain fields on mesh-like structures, which supports the idea of mesh/CAE-derived mechanical supervision.
- Static condensation and reduced stiffness/compliance matrices provide a classical mechanical analogy: reduce the structure to relationships between boundary/interface degrees of freedom.
- Topology optimization and load-path design commonly use compliance and strain energy density, supporting strain-energy-based mechanical attention as a useful signal.
