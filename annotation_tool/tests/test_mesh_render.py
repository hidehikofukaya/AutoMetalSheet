import numpy as np
import pyvista as pv

from annotation_tool.mesh_render import contact_zone_labels, detect_holes, feature_edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annulus(inner_r: float, outer_r: float, n: int = 32) -> pv.PolyData:
    """Flat annular mesh (a disc with a circular hole)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer_pts = np.column_stack([outer_r * np.cos(angles), outer_r * np.sin(angles), np.zeros(n)])
    inner_pts = np.column_stack([inner_r * np.cos(angles), inner_r * np.sin(angles), np.zeros(n)])
    pts = np.vstack([outer_pts, inner_pts])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([3, i, j, n + j])
        faces.extend([3, i, n + j, n + i])
    return pv.PolyData(pts, np.array(faces))


def _make_closed_solid_cylinder(radius: float = 5.0, height: float = 2.0, n: int = 32) -> pv.PolyData:
    """Water-tight solid cylinder (flat end caps + curved wall, no boundary edges).

    Simulates a *filled* cylindrical hole as produced by Mesh_Generater:
    - bottom cap : n triangles fanning from (0,0,0)
    - top cap    : n triangles fanning from (0,0,height)
    - wall       : 2n triangles connecting the two rings
    Feature edges at 30° exist at both cap-wall junctions (90° dihedral).
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bot_ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)])
    top_ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, height)])
    pts = np.vstack([bot_ring, top_ring, [[0.0, 0.0, 0.0]], [[0.0, 0.0, height]]])
    # point index layout:  0..n-1 = bot_ring,  n..2n-1 = top_ring,  2n = bot_center,  2n+1 = top_center
    bc, tc = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([3, bc, i, j])          # bottom cap fan
        faces.extend([3, tc, n + j, n + i])  # top cap fan (reversed winding → +z normal)
        faces.extend([3, i, n + i, n + j])   # wall tri 1
        faces.extend([3, i, n + j, j])       # wall tri 2
    return pv.PolyData(pts, np.array(faces))


def _make_elliptical_annulus(
    a_inner: float,
    b_inner: float,
    a_outer: float,
    b_outer: float,
    n: int = 32,
) -> pv.PolyData:
    """Flat annular mesh with an elliptical inner hole (semi-axes a, b)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer_pts = np.column_stack([a_outer * np.cos(angles), b_outer * np.sin(angles), np.zeros(n)])
    inner_pts = np.column_stack([a_inner * np.cos(angles), b_inner * np.sin(angles), np.zeros(n)])
    pts = np.vstack([outer_pts, inner_pts])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([3, i, j, n + j])
        faces.extend([3, i, n + j, n + i])
    return pv.PolyData(pts, np.array(faces))


def test_feature_edges_finds_all_cube_edges() -> None:
    cube = pv.Cube().triangulate()

    edges = feature_edges(cube)

    # a cube has exactly 12 sharp (90deg) edges and 8 corners -- well above
    # even the near-zero default threshold
    assert edges.n_cells == 12
    assert edges.n_points == 8


def test_feature_edges_hides_coplanar_triangulation_diagonal() -> None:
    # a single flat quad split into 2 triangles -- the shared diagonal is
    # perfectly coplanar (dihedral angle 0) and must not appear as an edge,
    # but the quad's 4 perimeter edges (boundary edges) always show
    quad = pv.PolyData(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        faces=np.hstack([[3, 0, 1, 2], [3, 0, 2, 3]]),
    )

    edges = feature_edges(quad)

    assert edges.n_cells == 4  # perimeter only, no diagonal
    assert edges.n_points == 4


def test_feature_edges_shows_dense_edges_on_curved_surface() -> None:
    # neighboring triangles on a curved surface are never perfectly
    # coplanar -- under the near-zero default threshold, every tessellation
    # edge on a sphere/cylinder is a genuine (if small) direction change
    sphere = pv.Sphere(theta_resolution=60, phi_resolution=60)

    edges = feature_edges(sphere)

    assert edges.n_cells > 0


def test_contact_zone_labels_no_adjacent() -> None:
    focus = pv.Cube().triangulate()

    labels = contact_zone_labels(focus, [])

    assert labels.shape == (focus.n_cells,)
    assert (labels == 0).all()


def test_contact_zone_labels_far_mesh_stays_zero() -> None:
    focus = pv.Cube().triangulate()
    far = pv.PolyData(np.array([[100.0, 0.0, 0.0]]))

    labels = contact_zone_labels(focus, [(far, 1)], threshold_mm=2.0)

    assert (labels == 0).all()


def test_contact_zone_labels_marks_close_cells() -> None:
    # cube spans [-0.5, 0.5] on each axis; right face at x=0.5
    # a point at (0.6, 0, 0) is 0.1mm from the right face
    focus = pv.Cube().triangulate()
    touching = pv.PolyData(np.array([[0.6, 0.0, 0.0]]))

    labels = contact_zone_labels(focus, [(touching, 1)], threshold_mm=1.0)

    assert (labels == 1).any()   # right-face cells labeled
    assert (labels == 0).any()   # opposite-face cells still 0


def test_contact_zone_labels_multiple_adjacent_distinct_labels() -> None:
    # Two flat planes 2 mm apart (x=0 and x=2).  Each adjacent mesh sits 0.3 mm
    # outside one plane, well beyond the other (distance ~2.3 mm > threshold).
    # Vertex-based labeling should assign label 1 to the right-plane cells and
    # label 2 to the left-plane cells with no cross-contamination.
    pts = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1],   # left plane  x=0
         [2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 0, 1]],  # right plane x=2
        dtype=float,
    )
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 2, 3], [3, 4, 5, 6], [3, 4, 6, 7]])
    focus = pv.PolyData(pts, faces)

    adj_right = pv.PolyData(np.array([[2.3, 0.5, 0.5]]))   # 0.3 mm from right plane
    adj_left  = pv.PolyData(np.array([[-0.3, 0.5, 0.5]]))  # 0.3 mm from left plane

    labels = contact_zone_labels(focus, [(adj_right, 1), (adj_left, 2)], threshold_mm=0.8)

    assert (labels == 1).any()   # right-plane cells labeled 1
    assert (labels == 2).any()   # left-plane cells labeled 2


# ---------------------------------------------------------------------------
# detect_holes
# ---------------------------------------------------------------------------

def test_detect_holes_no_holes_on_closed_mesh() -> None:
    # A closed sphere has no boundary edges and therefore no detectable holes
    sphere = pv.Sphere()
    holes = detect_holes(sphere)
    assert holes == []


def test_detect_holes_finds_inner_circle() -> None:
    # Annulus with inner radius 3 mm and outer radius 10 mm.
    # The inner loop (r≈3) should be detected; outer loop (r≈10) is excluded by
    # max_radius_mm=7 so we get exactly one result.
    annulus = _make_annulus(inner_r=3.0, outer_r=10.0)
    holes = detect_holes(annulus, min_radius_mm=1.0, max_radius_mm=7.0)

    assert len(holes) == 1
    h = holes[0]
    assert abs(h["radius_mm"] - 3.0) < 0.3         # radius accurate to 0.3 mm
    assert np.linalg.norm(h["center"]) < 0.3        # center near origin
    assert h["circularity"] > 0.90                  # clearly circular


def test_detect_holes_radius_filter() -> None:
    # With max_radius_mm < inner_r, nothing should be detected
    annulus = _make_annulus(inner_r=5.0, outer_r=15.0)
    assert detect_holes(annulus, min_radius_mm=1.0, max_radius_mm=4.0) == []


def test_detect_holes_normal_is_unit_vector() -> None:
    annulus = _make_annulus(inner_r=2.0, outer_r=8.0)
    holes = detect_holes(annulus, max_radius_mm=5.0)
    assert len(holes) == 1
    nrm = holes[0]["normal"]
    assert abs(np.linalg.norm(nrm) - 1.0) < 1e-6   # unit vector


def test_detect_holes_returns_shape_and_loop_points() -> None:
    # Every detected hole dict must carry the new 'shape' and 'loop_points' keys
    annulus = _make_annulus(inner_r=3.0, outer_r=10.0)
    holes = detect_holes(annulus, min_radius_mm=1.0, max_radius_mm=7.0)
    assert len(holes) == 1
    h = holes[0]
    assert "shape" in h
    assert "loop_points" in h
    # loop_points must be a 2-D array with 3 columns
    lp = h["loop_points"]
    assert isinstance(lp, np.ndarray)
    assert lp.ndim == 2 and lp.shape[1] == 3


def test_detect_holes_circle_classified_correctly() -> None:
    # A 32-point circle boundary should be classified as 'circle'
    annulus = _make_annulus(inner_r=4.0, outer_r=12.0)
    holes = detect_holes(annulus, min_radius_mm=1.0, max_radius_mm=8.0)
    assert len(holes) == 1
    assert holes[0]["shape"] == "circle"


def test_detect_holes_closed_mesh_feature_fallback() -> None:
    # _make_closed_solid_cylinder produces a water-tight mesh (no boundary edges).
    # The 90° dihedral at each cap-wall junction shows up as a feature edge ring.
    # detect_holes must fall back to feature-edge detection and deduplicate the
    # two cap-rim loops (one per end) into a single unique hole record.
    cyl = _make_closed_solid_cylinder(radius=5.0, height=2.0, n=32)

    # Verify the mesh is truly closed
    be = cyl.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                    feature_edges=False, manifold_edges=False)
    assert be.n_points == 0, "precondition: solid cylinder has no boundary edges"

    holes = detect_holes(cyl, min_radius_mm=1.0, max_radius_mm=10.0)
    assert len(holes) == 1, f"expected 1 hole after dedup, got {len(holes)}"
    h = holes[0]
    assert abs(h["radius_mm"] - 5.0) < 0.6   # equiv radius ≈ cylinder radius
    assert h["shape"] == "circle"
    assert abs(np.linalg.norm(h["normal"]) - 1.0) < 1e-6


def test_detect_holes_outer_perimeter_rejected() -> None:
    # A large solid cylinder (radius >> 30 mm threshold) whose end-cap loops
    # cover ~100% of the mesh's projected footprint must be classified as the
    # outer perimeter and rejected, returning no holes.
    cyl = _make_closed_solid_cylinder(radius=50.0, height=5.0, n=64)
    be = cyl.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                    feature_edges=False, manifold_edges=False)
    assert be.n_points == 0, "precondition: large solid cylinder has no boundary edges"
    holes = detect_holes(cyl, min_radius_mm=1.0, max_radius_mm=200.0)
    assert len(holes) == 0, (
        f"outer-perimeter end caps must be rejected, got {len(holes)} hole(s)"
    )


def test_detect_holes_slot_shape_detected() -> None:
    # Elliptical inner hole with 2:1 aspect ratio (a=8, b=4).
    # Isoperimetric quotient of an ellipse with a/b=2 is ≈ 0.80 → classified
    # as 'slot'.  Outer ellipse (a=25, b=20, equiv_r≈22 mm) is outside
    # max_radius_mm=12 so exactly one hole is returned.
    mesh = _make_elliptical_annulus(a_inner=8.0, b_inner=4.0, a_outer=25.0, b_outer=20.0)
    holes = detect_holes(mesh, min_radius_mm=1.0, max_radius_mm=12.0)
    assert len(holes) == 1
    h = holes[0]
    # Equivalent radius = sqrt(π·8·4 / π) = sqrt(32) ≈ 5.66 mm
    assert 4.0 < h["radius_mm"] < 8.0
    # Slot shape is less circular than a perfect circle
    assert h["circularity"] < 0.85
    assert h["shape"] == "slot"
    # New keys still present
    assert "loop_points" in h
