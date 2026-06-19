import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import skfmm
import os
import glob
import json
from skimage.morphology import medial_axis
from skimage.draw import disk
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from openpyxl import Workbook

# ==========================================
# CONFIGURATION
# ==========================================
MANUAL_PIXEL_SIZE = None  # Set to e.g., (0.5, 0.5) for manual pixel size in mm (x, y)

# Analysis parameters
FMM_WAYPOINT_INTERVAL_MM = 2.0  # Interval for waypoint sampling in FMM refinement
DIAMETER_SMOOTHING_SIGMA = 3.0  # Gaussian smoothing sigma for diameter profile
# Curvature is computed analytically from a smoothing cubic B-spline fit to the
# centerline (robust, scale-stable). SPLINE_SMOOTHING_FACTOR is the per-point
# residual budget in px²: lower hugs the centerline tighter (follows hairpins,
# more turn sensitivity), higher smooths more (but over-smoothing shortcuts sharp
# bends and merges real turns). 0.5 ≈ 0.7px RMS — tight enough to track tortuous
# veins while the guards below reject noise. INFLECTION_DEADBAND_FRAC ignores
# |curvature| below this fraction of the max when locating inflection points, so
# noise near a zero crossing doesn't spawn a spurious curve.
SPLINE_SMOOTHING_FACTOR = 0.5
# The data is a 2D projection of a 3D vessel: where the vessel curves toward/away
# from the camera the projection folds it back over itself into a near-hairpin,
# producing a physically impossible curvature spike (radius << vessel radius) that
# dominates max/std and the ∫|κ|ds total-turn integral. A vessel centerline cannot
# bend tighter than its own lumen, so we cap |κ| at 1/(factor · r_local), with
# r_local the local radius from the distance map. factor=1.0 is the hard physical
# floor (radius of curvature ≥ vessel radius); raise it to be more aggressive.
CURVATURE_CAP_RADIUS_FACTOR = 1.0
INFLECTION_DEADBAND_FRAC = 0.05
# A constant-sign arc must turn at least this many degrees to count as a real
# bend; smaller arcs are treated as straight, so noise on a near-straight vessel
# doesn't create phantom curves/inflections.
MIN_ARC_ANGLE_DEG = 20.0
CORRIDOR_RADIUS_FACTOR = 1.2  # Factor to expand corridor around skeleton for FMM
MIN_CORRIDOR_RADIUS = 1.5  # Minimum corridor radius in pixels
FMM_STEP_SIZE = 0.5  # Step size for FMM gradient descent
REPORT_PADDING = 40  # Padding around segmentation for report visualization


def get_report_path(nii_path):
    """Returns the report path for a given NIfTI file, handling .nii and .nii.gz."""
    if nii_path.endswith(".nii.gz"):
        return nii_path[:-7] + "_report.png"
    elif nii_path.endswith(".nii"):
        return nii_path[:-4] + "_report.png"
    else:
        return nii_path + "_report.png"


def case_from_nii(nii_path):
    """Case name (filename stem) for a mask, stripping .nii / .nii.gz."""
    base = os.path.basename(nii_path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return base


def _resolve_dataset(nii_path):
    """Map a mask to its dataset loader class and full-frame source image path.

    The dataset is inferred from the case-name prefix and its image directory /
    extension come from ``datasets.py`` (the single source of truth for these
    conventions). TJU lives at ``pre-data/TJU/{case}.png``; the side-by-side
    datasets use ``.jpg``.
    """
    import datasets

    case = case_from_nii(nii_path)
    if case.startswith("TJU"):
        cls, ext = datasets.TJUDataset, ".png"
    elif case.startswith("JAP"):
        cls, ext = datasets.JAPDataset, ".jpg"
    else:
        cls, ext = datasets.CurrentDataset, ".jpg"
    return cls, str(cls.image_dir / f"{case}{ext}")


def get_reference_path(nii_path):
    """Resolve the full-frame source image for a mask via the dataset conventions.

    Returns the *full-frame* source (which aligns with main.py's full-frame
    mask), not the cropped panel that the loaders' ``get_raw()`` would give for
    the side-by-side datasets.
    """
    return _resolve_dataset(nii_path)[1]


def get_reference_images(nii_path):
    """Return ``(raw_rgb, labeled_rgb)`` reference arrays for a mask, via datasets.py.

    ``raw_rgb`` is the full-frame source — it aligns with main.py's full-frame
    mask and is what the skeleton/clicks overlay. ``labeled_rgb`` is the
    clinician's traced version, shown in a pane beside the raw. For TJU these are
    two separate files (raw ``.png`` + ``{case} tracked.png``), loaded through the
    ``TJUDataset`` loader. For the side-by-side datasets the raw frame already
    holds the tracing in its left panel, so ``labeled_rgb`` is None (one pane
    suffices). Either element is None when its file is missing.
    """
    import datasets

    cls, path = _resolve_dataset(nii_path)
    if cls is datasets.TJUDataset:
        loader = cls(path)

        def _safe(fn):
            try:
                return fn()
            except (FileNotFoundError, OSError):
                return None

        raw, labeled = _safe(loader.get_raw), _safe(loader.get_labeled)
        if raw is None:
            print(f"No raw reference found at {loader.raw_path}")
        if labeled is not None:
            print(f"Loaded labeled reference: {loader.labeled_path}")
        else:
            print(f"No labeled reference found at {loader.labeled_path}")
        return raw, labeled

    # Side-by-side datasets: the full source frame already shows traced-left |
    # clean-right, so a single pane already puts the labeling next to the raw.
    if os.path.exists(path):
        print(f"Loaded reference image: {path}")
        return plt.imread(path), None
    print(f"No reference image found at {path}")
    return None, None


def get_clicks_path(nii_path):
    """Returns the path for saving/loading user click points."""
    if nii_path.endswith(".nii.gz"):
        return nii_path[:-7] + "_clicks.json"
    elif nii_path.endswith(".nii"):
        return nii_path[:-4] + "_clicks.json"
    else:
        return nii_path + "_clicks.json"


def load_and_preprocess(nii_path, manual_pixel_size=None):
    """Loads NIfTI file and extracts binary mask and pixel dimensions."""
    print(f"Loading {nii_path}...")
    img = nib.load(nii_path)
    data = np.asarray(img.dataobj)
    data_2d = np.squeeze(data)
    # binary_mask = np.flipud((data_2d > 0).T.astype(np.uint8))
    binary_mask = (data_2d > 0).T.astype(np.uint8)

    # Get pixel dimensions
    header = img.header
    zooms = header.get_zooms()[:3]
    if manual_pixel_size:
        ps_x, ps_y = manual_pixel_size
    else:
        ps_x, ps_y = zooms[0], zooms[1]

    print(f"Pixel Dimensions: {ps_x:.3f}mm x {ps_y:.3f}mm")
    return binary_mask, ps_x, ps_y


def skeletonize_and_graph(binary_mask, ps_x, ps_y):
    """Computes medial axis skeleton and builds a NetworkX graph."""
    print("Skeletonizing (Medial Axis)...")
    skeleton, dist_map = medial_axis(binary_mask, return_distance=True)
    y_coords, x_coords = np.where(skeleton)

    # Build graph from skeleton points
    g = nx.Graph()
    points = list(zip(x_coords, y_coords))
    points_set = set(points)  # O(1) lookup for neighbor checking
    for p in points:
        g.add_node(p)

    for p in points:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (p[0] + dx, p[1] + dy)
                if neighbor in points_set:
                    dist_mm = np.sqrt((dx * ps_x) ** 2 + (dy * ps_y) ** 2)
                    g.add_edge(p, neighbor, weight=dist_mm)
    return g, dist_map


def get_user_path(g, binary_mask, nii_path=None):
    """Allows user to manually select path points on an interactive plot.
    Loads saved clicks if available, otherwise opens interactive plot and saves clicks."""

    # Try to load saved clicks
    clicks_path = get_clicks_path(nii_path) if nii_path else None
    if clicks_path and os.path.exists(clicks_path):
        print(f"Loading saved click points from {clicks_path}...")
        with open(clicks_path, "r") as f:
            points = [tuple(p) for p in json.load(f)]
        print(f"Loaded {len(points)} saved click points.")
    else:
        print("Please select the path manually in the napari window...")

        import napari

        # Reference images via the dataset loaders: the raw full frame (under the
        # mask/skeleton, where you click) and the labeled tracing shown beside it.
        raw_img, labeled_img = (None, None)
        if nii_path is not None:
            raw_img, labeled_img = get_reference_images(nii_path)

        # Rasterize the skeleton nodes into an overlay image (cheap, vs drawing
        # thousands of graph edges as in the old matplotlib view).
        skeleton_img = np.zeros(binary_mask.shape, dtype=np.uint8)
        for x, y in g.nodes():
            skeleton_img[int(y), int(x)] = 1

        viewer = napari.Viewer(title="Select vein path — click points in order")
        if raw_img is not None:
            viewer.add_image(raw_img, name="raw")
        if labeled_img is not None:
            # Show the labeled tracing in its own pane just to the right of the
            # raw frame (napari translate is (row, col) = (y, x)).
            base_w = raw_img.shape[1] if raw_img is not None else binary_mask.shape[1]
            gap = max(10, base_w // 40)
            viewer.add_image(
                labeled_img, name="labeled (tracing)", translate=(0, base_w + gap)
            )
        # Explicit contrast_limits=(0, 1): these are binary {0, 1} images, so
        # without them napari may scale against 0..255 (or miss the sparse vessel
        # pixels when it subsamples a large frame) and render an all-black canvas.
        # Additive blending lets the vessel/skeleton glow over the reference (or
        # the black canvas) and keeps the 0 background transparent.
        viewer.add_image(
            binary_mask,
            name="mask",
            colormap="gray",
            contrast_limits=(0, 1),
            blending="additive",
        )
        viewer.add_image(
            skeleton_img,
            name="skeleton",
            colormap="cyan",
            contrast_limits=(0, 1),
            blending="additive",
            opacity=0.9,
        )
        pts_layer = viewer.add_points(
            name="endpoints", ndim=2, size=8, face_color="red"
        )
        pts_layer.mode = "add"
        viewer.reset_view()
        viewer.text_overlay.visible = True
        viewer.text_overlay.font_size = 10
        viewer.text_overlay.text = (
            "Select the vein path:\n"
            "1) On the 'endpoints' layer (add mode), click points in order\n"
            "   along the vein — start, any waypoints, then end\n"
            "2) Wrong point? select it and press Delete\n"
            "Close the window when done."
        )
        print(
            "napari opened. On the 'endpoints' layer (add mode), click the path "
            "points in order. Close the window when done."
        )
        napari.run()

        raw = np.asarray(pts_layer.data)
        if raw.size == 0:
            print("No points selected.")
            return []
        # napari Points are [row, col] = [y, x]; the graph nodes are (x, y).
        points = [(float(p[1]), float(p[0])) for p in raw]

        # Save clicks for future reuse (existing [x, y] format).
        if clicks_path:
            with open(clicks_path, "w") as f:
                json.dump([list(p) for p in points], f)
            print(f"Saved click points to {clicks_path}")

    print(f"Using {len(points)} click points. Calculating path...")

    # Map clicks to nearest graph nodes
    node_coords = np.array(g.nodes())
    tree = KDTree(node_coords)

    path_nodes = []
    for pt in points:
        # points are (x, y), matching the (x, y) graph node coordinates
        dist, idx = tree.query(pt)
        nearest_node = tuple(node_coords[idx])
        path_nodes.append(nearest_node)

    # Connect nodes
    full_path = [path_nodes[0]]
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        try:
            # Find shortest path on graph between clicked points
            segment = nx.shortest_path(g, u, v, weight="weight")
            # Append segment (excluding start node to avoid duplication)
            full_path.extend(segment[1:])
        except nx.NetworkXNoPath:
            print(f"No path between {u} and {v}")
            return []

    return full_path


def sample_waypoints_indices(path, interval_mm, ps_x, ps_y):
    """
    Sample waypoints along the path at roughly the specified interval.
    Returns INDICES into the path list.
    """
    if not path:
        return []

    indices = [0]
    accumulated_dist = 0

    for i in range(1, len(path)):
        prev = path[i - 1]
        curr = path[i]

        # Euclidean distance between consecutive pixels
        dist = np.sqrt(
            ((curr[0] - prev[0]) * ps_x) ** 2 + ((curr[1] - prev[1]) * ps_y) ** 2
        )
        accumulated_dist += dist

        if accumulated_dist >= interval_mm:
            indices.append(i)
            accumulated_dist = 0

    # Always include the end point index
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)

    return indices


def compute_fmm_segment(start_node, end_node, speed, mask):
    """Computes shortest path between two nodes using Fast Marching Method."""
    # Propagate travel time from start node
    Y, X = np.ogrid[: mask.shape[0], : mask.shape[1]]
    dist_from_start = np.sqrt((X - start_node[0]) ** 2 + (Y - start_node[1]) ** 2)
    phi = dist_from_start - 0.5
    phi = np.ma.MaskedArray(phi, ~mask.astype(bool))

    try:
        t_field = skfmm.travel_time(phi, speed)
    except ValueError:
        # Fallback if masking is too aggressive or start node is masked out
        return [start_node, end_node]

    # Backtrack via gradient descent
    path = [end_node]
    current = np.array(end_node, dtype=float)

    # Compute gradients (handle boundaries/inf)
    max_t = t_field.max()
    if np.ma.is_masked(max_t):
        max_t = 1e5
    t_grid = t_field.filled(max_t * 2.0)

    gy, gx = np.gradient(t_grid)

    step_size = FMM_STEP_SIZE
    max_steps = int(mask.size)

    for _ in range(max_steps):
        iy, ix = int(round(current[1])), int(round(current[0]))

        dist = np.sqrt(
            (current[0] - start_node[0]) ** 2 + (current[1] - start_node[1]) ** 2
        )
        if dist < 1.0:
            path.append(start_node)
            break

        if iy < 0 or iy >= t_grid.shape[0] or ix < 0 or ix >= t_grid.shape[1]:
            break

        dy = gy[iy, ix]
        dx = gx[iy, ix]

        mag = np.sqrt(dx**2 + dy**2)
        if mag == 0 or np.isnan(mag):
            break

        current[0] -= (dx / mag) * step_size
        current[1] -= (dy / mag) * step_size

        if np.isnan(current[0]) or np.isnan(current[1]):
            break

        path.append(tuple(current))

    return path[::-1]


def compute_centerline_fmm(binary_mask, longest_path_graph, dist_map, ps_x, ps_y):
    """Refines the centerline using FMM within a corridor around the graph path."""
    # Sample waypoints to follow topology
    indices = sample_waypoints_indices(
        longest_path_graph, interval_mm=FMM_WAYPOINT_INTERVAL_MM, ps_x=ps_x, ps_y=ps_y
    )
    print(
        f"Refining centerline using Fast Marching Method (Fluid-like) with {len(indices)} waypoints..."
    )

    # Speed map from distance transform
    speed = distance_transform_edt(binary_mask)

    full_path = []

    for i in range(len(indices) - 1):
        idx_start = indices[i]
        idx_end = indices[i + 1]

        start_node = longest_path_graph[idx_start]
        end_node = longest_path_graph[idx_end]

        # Create corridor mask around skeleton segment
        # Prevents merging adjacent loops
        skeleton_segment = longest_path_graph[idx_start : idx_end + 1]
        corridor_mask = np.zeros_like(binary_mask)
        for p in skeleton_segment:
            r = max(dist_map[p[1], p[0]] * CORRIDOR_RADIUS_FACTOR, MIN_CORRIDOR_RADIUS)
            rr, cc = disk((p[1], p[0]), r, shape=binary_mask.shape)
            corridor_mask[rr, cc] = 1

        local_mask = binary_mask * corridor_mask

        # Compute segment
        segment = compute_fmm_segment(start_node, end_node, speed, local_mask)

        # Add to full path (avoid duplicating the connection point)
        if i == 0:
            full_path.extend(segment)
        else:
            full_path.extend(segment[1:])

    return full_path


def fit_centerline_spline(longest_path, ps_x, ps_y, cumulative_dist):
    """Fit a smoothing cubic B-spline to the centerline.

    A parametric spline (x(t), y(t)) is fit in millimetre coordinates with a
    residual budget of ~SPLINE_SMOOTHING_FACTOR px² per point, so the fit tracks
    the vessel while ignoring pixel-level skeleton jitter. Curvature is then taken
    analytically from the spline derivatives,

        κ = (x'·y'' − y'·x'') / (x'² + y'²)^{3/2},

    which is parameterisation-invariant, so κ is true 1/mm regardless of the
    spline's internal parameter. Keeping the *sign* of κ (left vs right turn) is
    what lets ``count_curves`` find inflection points (sign changes).

    Returns ``(signed_curvature, fitted_xy)`` where ``signed_curvature`` is an
    array of length ``len(longest_path)`` aligned with ``cumulative_dist``, and
    ``fitted_xy`` is a densely sampled ``(M, 2)`` array of the fitted curve in
    *pixel* coordinates for overlaying on the report (None when no fit is made).
    On a too-short path or a failed fit, returns ``(zeros, None)`` so callers
    degrade gracefully.
    """
    path_arr = np.asarray(longest_path, dtype=float)
    n = len(path_arr)
    if n < 4:
        return np.zeros(n), None

    x_mm = path_arr[:, 0] * ps_x
    y_mm = path_arr[:, 1] * ps_y

    # Chord-length parameter in [0, 1]; the spline is fit *and* sampled on it, so
    # the returned curvature lines up with the original points (and cumulative_dist).
    total = cumulative_dist[-1]
    if total <= 0:
        return np.zeros(n), None
    t = cumulative_dist / total

    # splprep needs strictly increasing parameters: drop points coincident with
    # their predecessor (FMM can emit repeats), keeping the first occurrence.
    keep = np.concatenate(([True], np.diff(t) > 1e-9))
    t_fit, x_fit, y_fit = t[keep], x_mm[keep], y_mm[keep]
    if len(t_fit) < 4:
        return np.zeros(n), None

    ps_mean = 0.5 * (ps_x + ps_y)
    s = SPLINE_SMOOTHING_FACTOR * len(t_fit) * (ps_mean**2)
    try:
        tck, _ = splprep([x_fit, y_fit], u=t_fit, s=s, k=3)
        dx, dy = splev(t, tck, der=1)
        ddx, ddy = splev(t, tck, der=2)
        # Densely sample the fitted curve for a smooth overlay, back in pixels.
        t_dense = np.linspace(0.0, 1.0, max(n, 400))
        fx, fy = splev(t_dense, tck, der=0)
        fitted_xy = np.column_stack([np.asarray(fx) / ps_x, np.asarray(fy) / ps_y])
    except (ValueError, TypeError) as e:
        print(f"  spline curvature fit failed ({e}); falling back to zeros.")
        return np.zeros(n), None

    denom = np.power(dx**2 + dy**2, 1.5)
    signed_curvature = np.divide(
        dx * ddy - dy * ddx, denom, out=np.zeros_like(denom), where=denom > 1e-12
    )
    return signed_curvature, fitted_xy


def calculate_metrics(
    longest_path, binary_mask, ps_x, ps_y, max_length_mm
):
    """Computes geometric metrics (length, tortuosity, diameter, curvature) for the path.

    The diameter/volume calc recomputes a spacing-aware Euclidean distance transform
    internally (``sampling=(ps_y, ps_x)``) so radii come out in millimetres even when
    ``ps_x != ps_y``. It deliberately does *not* accept a precomputed map: a pixel-space
    one (e.g. ``medial_axis``'s output) would silently bias diameters and volume under
    anisotropic spacing.
    """
    # Convert to numpy for calculations
    path_arr = np.array(longest_path)

    # --- Path Distance ---
    diffs = np.diff(path_arr, axis=0)
    diffs_mm = np.sqrt((diffs[:, 0] * ps_x) ** 2 + (diffs[:, 1] * ps_y) ** 2)
    cumulative_dist = np.insert(np.cumsum(diffs_mm), 0, 0)

    # --- COMPUTE METRICS ---

    # --- Tortuosity ---
    start_node = longest_path[0]
    end_node = longest_path[-1]
    chord_length_mm = np.sqrt(
        ((start_node[0] - end_node[0]) * ps_x) ** 2
        + ((start_node[1] - end_node[1]) * ps_y) ** 2
    )
    tortuosity = max_length_mm / chord_length_mm if chord_length_mm > 0 else 1.0

    # --- Diameter Profile ---
    # Local vessel radius = Euclidean distance from each path point to the nearest
    # background pixel. It is computed as a *spacing-aware* distance transform so the
    # radii come out directly in millimetres even when ps_x != ps_y. The earlier code
    # used a pixel-space map (medial_axis or EDT without sampling) scaled by ps_x
    # only, which both assumed square pixels for the radius itself and then ignored
    # ps_y in the px->mm step -- so every diameter (and the volume) was silently
    # biased under anisotropic spacing. binary_mask is laid out (rows=Y, cols=X), so
    # the per-axis sampling is (ps_y, ps_x).
    dist_map_mm = distance_transform_edt(binary_mask, sampling=(ps_y, ps_x))
    path_radii_mm = np.array(
        [dist_map_mm[int(round(p[1])), int(round(p[0]))] for p in longest_path]
    )
    path_diameters_mm = path_radii_mm * 2.0

    # Smooth diameter profile
    path_diameters_mm = gaussian_filter1d(
        path_diameters_mm, sigma=DIAMETER_SMOOTHING_SIGMA
    )

    avg_diameter = np.mean(path_diameters_mm)
    max_diameter = np.max(path_diameters_mm)
    min_diameter = np.min(path_diameters_mm)
    std_diameter = np.std(path_diameters_mm)

    # --- Volume (Cylinders) ---
    segment_diameters = (path_diameters_mm[:-1] + path_diameters_mm[1:]) / 2
    segment_areas = np.pi * (segment_diameters / 2) ** 2
    volume_mm3 = np.sum(segment_areas * diffs_mm)

    # --- Curvature Profile (analytic, from a smoothing B-spline fit) ---
    # signed_curvature keeps the sign (turn direction) so curves can be counted by
    # inflection (sign change); curvature is its magnitude for the profile/stats.
    signed_curvature, spline_fit = fit_centerline_spline(
        longest_path, ps_x, ps_y, cumulative_dist
    )

    # Cap curvature at the local physical limit. path_radii_mm is the lumen radius
    # (in mm) from the spacing-aware distance map at each path point; the centerline
    # cannot bend tighter than 1/(factor · r) without the lumen self-intersecting,
    # so anything above that is a projection-fold artifact, not real geometry. Capping
    # the *signed* curvature keeps inflection signs intact for count_curves while
    # removing the spike from max/std and the ∫|κ|ds total-turn integral.
    radius_mm = path_radii_mm
    kappa_max = np.divide(
        1.0,
        CURVATURE_CAP_RADIUS_FACTOR * radius_mm,
        out=np.full_like(radius_mm, np.inf),
        where=radius_mm > 0,
    )
    signed_curvature = np.clip(signed_curvature, -kappa_max, kappa_max)

    curvature = np.abs(signed_curvature)

    if np.any(curvature):
        max_k_idx = int(np.argmax(curvature))
        max_k_point = longest_path[max_k_idx]
        max_k_val = float(curvature[max_k_idx])
        avg_curvature = float(np.mean(curvature))
        std_curvature = float(np.std(curvature))
    else:
        max_k_idx = 0
        max_k_point = longest_path[0]
        max_k_val = 0.0
        avg_curvature = 0.0
        std_curvature = 0.0

    metrics = {
        "length": max_length_mm,
        "tortuosity": tortuosity,
        "avg_diameter": avg_diameter,
        "min_diameter": min_diameter,
        "max_diameter": max_diameter,
        "std_diameter": std_diameter,
        "volume": volume_mm3,
        "avg_curvature": avg_curvature,
        "max_curvature": max_k_val,
        "std_curvature": std_curvature,
        "start_node": start_node,
        "end_node": end_node,
        "max_k_point": max_k_point,
        "max_k_idx": max_k_idx,
        "max_k_val": max_k_val,
        "spline_fit": spline_fit,
    }

    return (
        metrics,
        cumulative_dist,
        path_diameters_mm,
        curvature,
        dist_map_mm,
        signed_curvature,
    )


def print_summary(metrics):
    """Prints a text summary of the calculated metrics."""
    print("\n" + "=" * 30)
    print("       VEIN ANALYSIS SUMMARY       ")
    print("=" * 30)
    print(f"Length:          {metrics['length']:.2f} mm")
    print(f"Tortuosity:      {metrics['tortuosity']:.2f}")
    print(f"Avg Diameter:    {metrics['avg_diameter']:.2f} mm")
    print(f"Min Diameter:    {metrics['min_diameter']:.2f} mm")
    print(f"Max Diameter:    {metrics['max_diameter']:.2f} mm")
    print(f"Std Diameter:    {metrics['std_diameter']:.2f} mm")
    print(f"Est. Volume:     {metrics['volume']:.2f} mm³")
    print(f"Avg Curvature:   {metrics['avg_curvature']:.4f} mm⁻¹")
    print(f"Max Curvature:   {metrics['max_curvature']:.4f} mm⁻¹")
    print(f"Std Curvature:   {metrics['std_curvature']:.4f} mm⁻¹")
    if "inflection_count" in metrics:
        print(f"Inflections:     {metrics['inflection_count']}")
        print(f"Total Turning:   {metrics['total_curvature']:.1f}°")
    if "turns_90" in metrics:
        print(f"Turns ≥90°:      {metrics['turns_90']}")
        print(f"Turns ≥180°:     {metrics['turns_180']}")
        print(f"Turns ≥270°:     {metrics['turns_270']}")
    print("=" * 30 + "\n")


def turn_color(angle_deg):
    """Severity color for a turn of the given cumulative angle (degrees)."""
    if angle_deg >= 270:
        return "darkred"
    if angle_deg >= 180:
        return "red"
    return "orange"


def turn_label(angle_deg):
    """Legend label bucketing a turn by the strictest threshold it meets."""
    if angle_deg >= 270:
        return "Turn ≥270°"
    if angle_deg >= 180:
        return "Turn ≥180°"
    return "Turn ≥90°"


def count_curves(signed_curvature, cumulative_dist):
    """Count the vein's curves from its signed-curvature profile.

    The centerline is split at its inflection points (curvature sign changes);
    each constant-sign arc between inflections is one bend, and that bend's turn
    angle is the integral of curvature over it, ∫κ ds (radians -> degrees). This
    is the curvature/spline analogue of the older heading-change method and lines
    up with how vascular-tortuosity work quantifies turns (inflection-count and
    sum-of-angles metrics). Tiny ripples are ignored via a deadband on |κ| (a
    fraction of the max) so noise near a zero crossing can't spawn a spurious
    inflection.

    Counts are cumulative: a 200° bend counts as both a ≥90° and a ≥180° turn.

    Returns ``(turns_90, turns_180, turns_270, turn_events, inflection_count,
    total_curvature_deg)`` where ``turn_events`` is ``(path_index, angle_deg)`` per
    bend ≥90° (located at its sharpest point) and ``total_curvature_deg`` is the
    total absolute turning ∫|κ| ds over the whole vein.
    """
    signed_curvature = np.asarray(signed_curvature, dtype=float)
    n = len(signed_curvature)
    if n < 3 or not np.any(signed_curvature):
        return 0, 0, 0, [], 0, 0.0

    abs_k = np.abs(signed_curvature)
    total_curvature_deg = float(np.degrees(np.trapezoid(abs_k, cumulative_dist)))

    # Classify each point's turn direction, treating near-zero curvature as
    # "straight" (0) so faint ripples don't register as inflections.
    deadband = INFLECTION_DEADBAND_FRAC * abs_k.max()
    sign = np.sign(signed_curvature)
    sign[abs_k < deadband] = 0

    # Group into maximal constant-sign arcs. A straight (0) run is folded into the
    # preceding arc; a new arc begins only at a genuine sign reversal, so the
    # number of arcs minus one is the inflection count.
    segments = []  # (lo, hi) inclusive index ranges
    cur_sign = 0
    seg_lo = 0
    for i, s in enumerate(sign):
        if s == 0:
            continue
        if cur_sign == 0:
            cur_sign, seg_lo = s, i
        elif s != cur_sign:
            segments.append((seg_lo, i - 1))
            cur_sign, seg_lo = s, i
    if cur_sign != 0:
        segments.append((seg_lo, n - 1))

    # Reduce each arc to its signed turning angle, keeping only arcs that bend
    # enough to be a real curve (drops near-straight/noise runs).
    arcs = []  # (location index of sharpest point, signed angle in degrees)
    for lo, hi in segments:
        angle = np.degrees(
            np.trapezoid(signed_curvature[lo : hi + 1], cumulative_dist[lo : hi + 1])
        )
        if abs(angle) >= MIN_ARC_ANGLE_DEG:
            loc = lo + int(np.argmax(abs_k[lo : hi + 1]))
            arcs.append((loc, float(angle)))

    # Inflections = sign reversals between consecutive significant arcs.
    inflection_count = sum(
        1 for (_, a), (_, b) in zip(arcs, arcs[1:]) if np.sign(a) != np.sign(b)
    )

    turns_90 = turns_180 = turns_270 = 0
    turn_events = []  # (path index of sharpest point, bend angle in degrees)
    for loc, angle in arcs:
        mag = abs(angle)
        if mag >= 270:
            turns_270 += 1
        if mag >= 180:
            turns_180 += 1
        if mag >= 90:
            turns_90 += 1
            turn_events.append((loc, mag))

    return (
        turns_90,
        turns_180,
        turns_270,
        turn_events,
        inflection_count,
        total_curvature_deg,
    )


def export_all_metrics_xlsx(all_metrics, output_path):
    """Exports metrics from all processed files to a single XLSX file."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Vein Metrics"

    headers = [
        "File",
        "Length (mm)",
        "Tortuosity",
        "Avg Diameter (mm)",
        "Min Diameter (mm)",
        "Max Diameter (mm)",
        "Std Diameter (mm)",
        "Volume (mm³)",
        "Avg Curvature (1/mm)",
        "Max Curvature (1/mm)",
        "Std Curvature (1/mm)",
        "Inflections",
        "Total Turning (°)",
        "Turns ≥90°",
        "Turns ≥180°",
        "Turns ≥270°",
    ]
    ws.append(headers)

    for nii_path, metrics in all_metrics:
        row = [
            os.path.basename(nii_path),
            round(metrics["length"], 2),
            round(metrics["tortuosity"], 2),
            round(metrics["avg_diameter"], 2),
            round(metrics["min_diameter"], 2),
            round(metrics["max_diameter"], 2),
            round(metrics["std_diameter"], 2),
            round(metrics["volume"], 2),
            round(metrics["avg_curvature"], 4),
            round(metrics["max_curvature"], 4),
            round(metrics["std_curvature"], 4),
            metrics.get("inflection_count", 0),
            round(metrics.get("total_curvature", 0.0), 1),
            metrics["turns_90"],
            metrics["turns_180"],
            metrics["turns_270"],
        ]
        ws.append(row)

    # Auto-size columns
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 2

    wb.save(output_path)
    print(f"Metrics saved to {output_path}")


def visualize_results(
    binary_mask,
    dist_map,
    longest_path,
    cumulative_dist,
    path_diameters_mm,
    curvature,
    metrics,
    filename,
):
    """Generates and saves a comprehensive visualization report."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Vein Analysis Report: {filename}", fontsize=16, fontweight="bold")

    # Layout: Anatomy on top, stats below
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # Panel 1: Anatomy & Centerline
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(binary_mask, cmap="gray")

    path_x_plot, path_y_plot = zip(*longest_path)
    ax1.plot(
        path_x_plot,
        path_y_plot,
        "r-",
        linewidth=2.5,
        alpha=0.4,
        label="Centerline (raw)",
    )

    # Overlay the smoothing B-spline the curvature/turns are measured from, so the
    # fit can be eyeballed against the raw centerline it was computed from.
    spline_fit = metrics.get("spline_fit")
    if spline_fit is not None:
        ax1.plot(
            spline_fit[:, 0],
            spline_fit[:, 1],
            color="cyan",
            linewidth=1.5,
            alpha=0.95,
            label="Spline fit",
        )

    # Markers
    ax1.scatter(
        metrics["start_node"][0],
        metrics["start_node"][1],
        c="lime",
        s=100,
        zorder=5,
        edgecolors="black",
        label="Start",
    )
    ax1.scatter(
        metrics["end_node"][0],
        metrics["end_node"][1],
        c="magenta",
        s=100,
        zorder=5,
        edgecolors="black",
        label="End",
    )
    ax1.scatter(
        metrics["max_k_point"][0],
        metrics["max_k_point"][1],
        c="yellow",
        marker="x",
        s=150,
        zorder=5,
        linewidth=3,
        label="Max Bend",
    )

    # Mark each detected turn on the centerline, colored by severity. One legend
    # entry per severity bucket so repeated turns don't clutter the legend.
    turn_events = metrics.get("turn_events", [])
    labeled = set()
    for path_idx, angle_deg in turn_events:
        if path_idx >= len(longest_path):
            continue
        tx, ty = longest_path[path_idx]
        label = turn_label(angle_deg)
        ax1.scatter(
            tx,
            ty,
            facecolors="none",
            edgecolors=turn_color(angle_deg),
            s=180,
            linewidths=2.5,
            zorder=6,
            label=label if label not in labeled else None,
        )
        labeled.add(label)

    ax1.set_title("Vein Anatomy & Centerline")
    ax1.legend(loc="upper right", fontsize="small")

    # Crop view to the segmentation
    y_indices, x_indices = np.where(binary_mask > 0)
    if len(y_indices) > 0 and len(x_indices) > 0:
        pad = REPORT_PADDING
        y_min = max(0, y_indices.min() - pad)
        y_max = min(binary_mask.shape[0], y_indices.max() + pad)
        x_min = max(0, x_indices.min() - pad)
        x_max = min(binary_mask.shape[1], x_indices.max() + pad)

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_max, y_min)

    # Add global stats
    global_stats = (
        f"Length:      {metrics['length']:.2f} mm\n"
        f"Tortuosity:  {metrics['tortuosity']:.2f}\n"
        f"Est. Volume: {metrics['volume']:.2f} mm³"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    ax1.text(
        0.02,
        0.05,
        global_stats,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        bbox=props,
        family="monospace",
    )

    # Panel 2: Diameter Profile
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(cumulative_dist, path_diameters_mm, color="tab:blue", linewidth=2)
    ax2.fill_between(cumulative_dist, path_diameters_mm, color="tab:blue", alpha=0.1)

    ax2.scatter(
        cumulative_dist[0], path_diameters_mm[0], c="lime", edgecolors="black", zorder=5
    )
    ax2.scatter(
        cumulative_dist[-1],
        path_diameters_mm[-1],
        c="magenta",
        edgecolors="black",
        zorder=5,
    )

    ax2.axhline(
        metrics["avg_diameter"],
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"Mean: {metrics['avg_diameter']:.2f}mm",
    )
    ax2.set_title("Diameter Profile")
    ax2.set_xlabel("Distance along vein (mm)")
    ax2.set_ylabel("Diameter (mm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Diameter Stats
    diam_stats = (
        f"Avg: {metrics['avg_diameter']:.2f} mm\n"
        f"Min: {metrics['min_diameter']:.2f} mm\n"
        f"Max: {metrics['max_diameter']:.2f} mm\n"
        f"Std: {metrics['std_diameter']:.2f} mm"
    )
    ax2.text(
        0.02,
        0.95,
        diam_stats,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
        family="monospace",
    )

    # Panel 3: Curvature Profile
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(cumulative_dist, curvature, color="tab:orange", linewidth=2)
    ax3.fill_between(cumulative_dist, curvature, color="tab:orange", alpha=0.1)

    ax3.scatter(
        cumulative_dist[0], curvature[0], c="lime", edgecolors="black", zorder=5
    )
    ax3.scatter(
        cumulative_dist[-1], curvature[-1], c="magenta", edgecolors="black", zorder=5
    )
    ax3.scatter(
        cumulative_dist[metrics["max_k_idx"]],
        metrics["max_k_val"],
        c="yellow",
        marker="x",
        s=100,
        linewidth=2.5,
        zorder=5,
        label="Max Bend",
    )

    # Mark each turn at its arc-length position, labeled with its angle.
    for path_idx, angle_deg in turn_events:
        if path_idx >= len(cumulative_dist):
            continue
        color = turn_color(angle_deg)
        ax3.axvline(
            cumulative_dist[path_idx],
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )
        ax3.text(
            cumulative_dist[path_idx],
            0.98,
            f"{angle_deg:.0f}°",
            transform=ax3.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=color,
        )

    ax3.set_title("Curvature Profile")
    ax3.set_xlabel("Distance along vein (mm)")
    ax3.set_ylabel("Curvature (1/mm)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    # Curvature Stats
    curv_stats = (
        f"Tortuosity: {metrics['tortuosity']:.2f}\n"
        f"Avg: {metrics['avg_curvature']:.4f} mm⁻¹\n"
        f"Max: {metrics['max_curvature']:.4f} mm⁻¹\n"
        f"Std: {metrics['std_curvature']:.4f} mm⁻¹\n"
        f"Inflections: {metrics.get('inflection_count', 0)}   "
        f"Total turn: {metrics.get('total_curvature', 0.0):.0f}°\n"
        f"Turns: {metrics['turns_90']} (≥90°)  "
        f"{metrics['turns_180']} (≥180°)  {metrics['turns_270']} (≥270°)"
    )
    ax3.text(
        0.02,
        0.95,
        curv_stats,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
        family="monospace",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    plt.savefig(get_report_path(filename), dpi=300)
    plt.close()


def analyze_vein(nii_path):
    """Main function to orchestrate the vein analysis pipeline."""
    binary_mask, ps_x, ps_y = load_and_preprocess(nii_path, MANUAL_PIXEL_SIZE)
    g, dist_map = skeletonize_and_graph(binary_mask, ps_x, ps_y)

    # Get path from user
    longest_path_graph = get_user_path(g, binary_mask, nii_path)

    if not longest_path_graph:
        print("No path found.")
        return

    # Refine path with FMM (follows loops)
    longest_path = compute_centerline_fmm(
        binary_mask, longest_path_graph, dist_map, ps_x, ps_y
    )

    # Recalculate length for FMM path
    path_arr = np.array(longest_path)
    diffs = np.diff(path_arr, axis=0)
    diffs_mm = np.sqrt((diffs[:, 0] * ps_x) ** 2 + (diffs[:, 1] * ps_y) ** 2)
    max_length_mm = np.sum(diffs_mm)

    (
        metrics,
        cumulative_dist,
        path_diameters_mm,
        curvature,
        dist_map_mm,
        signed_curvature,
    ) = calculate_metrics(
        longest_path, binary_mask, ps_x, ps_y, max_length_mm
    )

    # Count curves from the spline curvature: bends between inflection points.
    turns_90, turns_180, turns_270, turn_events, inflections, total_curv = count_curves(
        signed_curvature, cumulative_dist
    )
    metrics["turns_90"] = turns_90
    metrics["turns_180"] = turns_180
    metrics["turns_270"] = turns_270
    metrics["turn_events"] = turn_events
    metrics["inflection_count"] = inflections
    metrics["total_curvature"] = total_curv

    print_summary(metrics)
    visualize_results(
        binary_mask,
        dist_map_mm,
        longest_path,
        cumulative_dist,
        path_diameters_mm,
        curvature,
        metrics,
        nii_path,
    )

    return metrics


if __name__ == "__main__":
    data_dir = "data"
    nii_files = glob.glob(os.path.join(data_dir, "*.nii")) + glob.glob(
        os.path.join(data_dir, "*.nii.gz")
    )

    all_metrics = []
    for nii_path in nii_files:
        report_path = get_report_path(nii_path)
        if os.path.exists(report_path):
            print(f"Skipping {nii_path} (Report already exists)")
            continue

        print(f"Processing {nii_path}...")
        metrics = analyze_vein(nii_path)
        if metrics is not None:
            all_metrics.append((nii_path, metrics))

    if all_metrics:
        export_all_metrics_xlsx(all_metrics, os.path.join(data_dir, "metrics.xlsx"))
