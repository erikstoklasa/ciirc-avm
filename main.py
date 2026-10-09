import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import skfmm
import os
import re
import glob
import json
import textwrap
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from skimage.morphology import medial_axis
from skimage.draw import disk, line
from skimage.graph import MCP_Geometric
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.ndimage import label as ndimage_label
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from scipy.stats import kurtosis, skew
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from openpyxl import Workbook

import datasets
from spacing import PLAUSIBLE_MM_PER_PX

# ==========================================
# CONFIGURATION
# ==========================================
MANUAL_PIXEL_SIZE = None  # Set to e.g., (0.5, 0.5) for manual pixel size in mm (x, y)

# Analysis parameters
MEDIAL_AXIS_SEED = 0  # Seed for medial_axis tie-breaking (reproducible skeletons)
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
# Shape features (diameter distribution): the distance transform falls off toward
# the mask tips, so this many median diameters are trimmed from each end of the
# profile before taking percentiles/moments. DILATION_RATIO marks a segment as
# dilated when its diameter exceeds this multiple of the vein's median.
DIAMETER_END_TRIM_DIAMETERS = 1.0
MIN_FEATURE_SAMPLES = 10  # fewer samples after trimming -> use the whole profile
DILATION_RATIO = 1.5
CORRIDOR_RADIUS_FACTOR = 1.2  # Factor to expand corridor around skeleton for FMM
MIN_CORRIDOR_RADIUS = 1.5  # Minimum corridor radius in pixels
# Start/end clicks link to skeleton nodes within (nearest-node distance + this many
# local vessel radii) that they can reach by a straight line inside the mask.
ENDPOINT_ATTACH_RADII = 2.0
# After FMM, each end is carried on along its direction (taken over the last
# EXTEND_TANGENT_RADII typical vessel radii) until it hits the mask wall; ends with
# no wall within EXTEND_MAX_RADII radii (a mid-vessel click) are left at the click.
EXTEND_TANGENT_RADII = 2.0
EXTEND_MAX_RADII = 4.0
FMM_STEP_SIZE = 0.5  # Step size for FMM gradient descent
REPORT_PADDING = 40  # Padding around segmentation for report visualization
FLOW_ARROW_COUNT = 8  # Base direction arrows, evenly spaced by arc length (flow diagram)
# Extra arrows in bends: one more arrow per this many degrees of accumulated turning
# (∫|κ|ds), so hairpins and tight loops get denser arrows than straight runs.
FLOW_ARROW_TURN_DEG = 60.0
FLOW_ARROW_MIN_GAP_PX = 8.0  # Minimum distance between arrows so they don't pile up
FLOW_CMAP = "plasma"  # Colormap encoding distance along the vein (start -> end)
SCALE_BAR_STEPS_MM = (0.5, 1, 2, 5, 10, 20, 50)  # "Nice" scale-bar lengths to pick from

# QC report ({case}_qc.png, bundled into data/qc_report.pdf)
QC_SUFFIX = "_qc.png"
QC_DPI = 110  # keeps a 45-case PDF reasonably small while width lines stay legible
QC_WIDTH_INTERVAL_MM = 2.0  # Draw a diameter width line every this many mm...
QC_MAX_WIDTH_LINES = 30  # ...but no more than this many on long veins
# Automatic-check thresholds (a hit adds a flag; nothing is rejected)
QC_CLICK_SNAP_WARN_MM = 1.0  # click farther than this from the path
QC_MIN_DIAMETER_PX = 3.0  # min diameter narrower than this many (mask-grid) px
QC_CURV_CAPPED_WARN_FRAC = 0.05  # share of curvature samples hitting the physical cap
QC_SPACING_MISMATCH_FRAC = 0.02  # NIfTI header vs spacing cache disagreement
QC_MIN_PIECE_PX = 20  # extra mask pieces smaller than this are specks, not flagged


def _strip_nii_suffix(nii_path):
    """Drop a trailing .nii / .nii.gz extension (single source of this rule)."""
    if nii_path.endswith(".nii.gz"):
        return nii_path[:-7]
    if nii_path.endswith(".nii"):
        return nii_path[:-4]
    return nii_path


def get_report_path(nii_path):
    """Returns the QC report path for a given NIfTI file, handling .nii and .nii.gz.

    Its existence also marks a case as processed (main.py / pipeline.py --next).
    """
    return _strip_nii_suffix(nii_path) + QC_SUFFIX


def get_spacing_path(nii_path):
    """Returns the cached spacing path (written by pipeline.py) for a NIfTI file."""
    return _strip_nii_suffix(nii_path) + "_spacing.json"


def load_scale_calibration(nii_path):
    """Return the cached ``{case}_spacing.json`` dict, or None if absent/unreadable.

    Of interest here: ``endpoints`` (the two clicked ends of the image's own scale
    bar, native-pixel (row, col)) and ``value_mm`` (its typed physical length).
    """
    try:
        with open(get_spacing_path(nii_path), "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def get_clicks_path(nii_path):
    """Returns the path for saving/loading user click points."""
    return _strip_nii_suffix(nii_path) + "_clicks.json"


def case_from_nii(nii_path):
    """Case name (filename stem) for a mask, stripping .nii / .nii.gz."""
    return os.path.basename(_strip_nii_suffix(nii_path))


def _resolve_dataset(nii_path):
    """Map a mask to its dataset loader class and full-frame source image path.

    The dataset is inferred from the case-name prefix and its image directory /
    extension come from ``datasets.py`` (the single source of truth for these
    conventions). TJU lives at ``pre-data/TJU/{case}.png``; the side-by-side
    datasets use ``.jpg``.
    """
    case = case_from_nii(nii_path)
    if case.startswith("TJU"):
        cls, ext = datasets.TJUDataset, ".png"
    elif case.startswith("JAP"):
        cls, ext = datasets.JAPDataset, ".jpg"
    else:
        cls, ext = datasets.CurrentDataset, ".jpg"
    return cls, str(cls.image_dir / f"{case}{ext}")


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


def load_and_preprocess(nii_path, manual_pixel_size=None):
    """Loads NIfTI file and extracts binary mask and pixel dimensions."""
    print(f"Loading {nii_path}...")
    img = nib.load(nii_path)
    data = np.asarray(img.dataobj)
    data_2d = np.squeeze(data)
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
    # medial_axis breaks ties randomly; a fixed seed keeps the skeleton (and so the
    # path and every metric) identical across runs.
    skeleton, dist_map = medial_axis(
        binary_mask, return_distance=True, rng=MEDIAL_AXIS_SEED
    )
    y_coords, x_coords = np.where(skeleton)

    # Build graph from skeleton points
    g = nx.Graph(ps_x=ps_x, ps_y=ps_y)
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


def get_user_path(g, binary_mask, dist_map, nii_path=None):
    """Allows user to manually select path points on an interactive plot.
    Loads saved clicks if available, otherwise opens interactive plot and saves clicks.

    Returns ``(full_path, points, snap_px)``: the graph path, the clicked (x, y)
    points, and each click's offset (px) from the path (see ``route_clicks``) —
    the last two feed the QC page. ``full_path`` is empty when no path was found.
    """

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

        # The mask/skeleton live on the saved nii grid, which the pipeline may have
        # upsampled (e.g. --upscale 2), while the reference images load at native
        # resolution. Scale the references up to the mask's pixel coordinates so the
        # skeleton overlays the raw frame instead of being shifted/stretched off to
        # the side. The mask/skeleton stay at scale 1 (the grid clicks map into).
        ref_scale = (1.0, 1.0)
        if raw_img is not None:
            ref_scale = (
                binary_mask.shape[0] / raw_img.shape[0],
                binary_mask.shape[1] / raw_img.shape[1],
            )

        viewer = napari.Viewer(title="Select vein path — click points in order")
        if raw_img is not None:
            viewer.add_image(raw_img, name="raw", scale=ref_scale)
        if labeled_img is not None:
            # Show the labeled tracing in its own pane just to the right of the raw
            # frame, in the same scaled coordinates (napari translate is world-space
            # (row, col) = (y, x), applied after scale).
            base_w = (
                raw_img.shape[1] * ref_scale[1]
                if raw_img is not None
                else binary_mask.shape[1]
            )
            gap = max(10, base_w // 40)
            viewer.add_image(
                labeled_img,
                name="labeled (tracing)",
                scale=ref_scale,
                translate=(0, base_w + gap),
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
            return [], [], []
        # napari Points are [row, col] = [y, x]; the graph nodes are (x, y).
        points = [(float(p[1]), float(p[0])) for p in raw]

        # Save clicks for future reuse (existing [x, y] format).
        if clicks_path:
            with open(clicks_path, "w") as f:
                json.dump([list(p) for p in points], f)
            print(f"Saved click points to {clicks_path}")

    print(f"Using {len(points)} click points. Calculating path...")
    return route_clicks(g, binary_mask, dist_map, points)


def _attach_endpoint(g, tree, node_coords, binary_mask, dist_map, pt):
    """Add the clicked start/end ``pt`` to ``g`` as a path node of its own.

    The medial axis stops short of the vessel tips (and can sit off-centre in
    blobs), so snapping a tip click to its nearest skeleton node loses the end of
    the vessel. Instead the click (moved onto the nearest mask pixel if it landed
    just outside) becomes a node, linked to every nearby skeleton node by a
    centre-seeking geodesic: a minimum-cost route through the mask with step cost
    1 / (distance to the wall), so it bends with the vessel and is pulled toward
    its centre, like the FMM refinement. Each edge stores its pixel route
    (``pixels``) and is weighted by that route's length in mm, so the
    shortest-path search picks the attachment that best continues toward the
    other click (which also skips medial-axis spurs near the tip).

    Returns ``(node, off_px)``: the node and how far the click was moved (px).
    """
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = binary_mask.shape
    x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
    if not binary_mask[y, x]:
        my, mx = np.nonzero(binary_mask)
        k = np.argmin((mx - pt[0]) ** 2 + (my - pt[1]) ** 2)
        x, y = int(mx[k]), int(my[k])
    node = (x, y)
    off_px = float(np.hypot(pt[0] - x, pt[1] - y))
    if node in g:
        return node, off_px

    d0, idx = tree.query(node)
    nearest = tuple(node_coords[idx])
    radius = d0 + ENDPOINT_ATTACH_RADII * max(dist_map[nearest[1], nearest[0]], 1.0)
    candidates = [tuple(node_coords[i]) for i in tree.query_ball_point(node, radius)]

    # Geodesics within a window around the click (all candidates lie inside it).
    r = int(np.ceil(radius)) + 2
    y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, h), max(x - r, 0), min(x + r + 1, w)
    win_mask = binary_mask[y0:y1, x0:x1].astype(bool)
    costs = np.full(win_mask.shape, np.inf)
    costs[win_mask] = 1.0 / np.maximum(dist_map[y0:y1, x0:x1][win_mask], 1.0)
    mcp = MCP_Geometric(costs)
    cum, _ = mcp.find_costs([(y - y0, x - x0)])

    ps = np.array([g.graph["ps_x"], g.graph["ps_y"]])
    for c in candidates:
        if not np.isfinite(cum[c[1] - y0, c[0] - x0]):
            continue  # not connected to the click inside the window
        route = [(cc + x0, rr + y0) for rr, cc in mcp.traceback((c[1] - y0, c[0] - x0))]
        length_mm = float(np.sum(np.hypot(*(np.diff(route, axis=0) * ps).T)))
        g.add_edge(node, c, weight=length_mm, pixels=route)
    if g.degree(node) == 0:
        # Not connected within the window (odd mask shape): straight to the nearest node.
        rr, cc = line(y, x, nearest[1], nearest[0])
        route = [(int(c), int(r)) for r, c in zip(rr, cc)]
        g.add_edge(node, nearest, weight=float(np.hypot(*((np.subtract(nearest, node)) * ps))),
                   pixels=route)
    return node, off_px


def _expand_connectors(g, path):
    """Replace each endpoint-connector edge in ``path`` by its stored pixel route."""
    out = [path[0]]
    for p, q in zip(path, path[1:]):
        route = g.edges[p, q].get("pixels")
        if route is not None:
            if route[0] != p:
                route = route[::-1]
            out.extend(route[1:-1])
        out.append(q)
    return out


def extend_to_wall(path, binary_mask, dist_map):
    """Continue both ends of the centerline straight on until they leave the mask.

    The clicks rarely sit exactly on the vessel's tips, so the path is carried on
    along its end direction (averaged over the last couple of vessel radii) until it
    hits the segmentation wall. An end is left as is when no wall is reached within
    ``EXTEND_MAX_RADII`` vessel radii (typical or local at that end, whichever is
    larger): the click was then mid-vessel, marking
    where the analysed stretch should stop.
    """
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return list(map(tuple, pts))
    h, w = binary_mask.shape
    ij = np.clip(np.round(pts).astype(int), 0, [w - 1, h - 1])
    radius = max(float(np.median(dist_map[ij[:, 1], ij[:, 0]])), 1.0)

    def ray(end_pts):
        # end_pts runs from the tip inward
        tip = end_pts[0]
        seg = np.cumsum(np.hypot(*np.diff(end_pts, axis=0).T))
        k = min(int(np.searchsorted(seg, max(EXTEND_TANGENT_RADII * radius, 3.0))) + 1,
                len(end_pts) - 1)
        direction = tip - end_pts[k]
        norm = np.hypot(*direction)
        if norm == 0:
            return []
        direction /= norm
        # Reach scales with the wider of the vessel's typical and the end's local
        # radius, so blunt, swollen tips are still reached.
        e = np.clip(np.round(end_pts[: k + 1]).astype(int), 0, [w - 1, h - 1])
        reach = EXTEND_MAX_RADII * max(radius, float(dist_map[e[:, 1], e[:, 0]].max()))
        out = []
        for i in range(1, int(reach / FMM_STEP_SIZE) + 1):
            q = tip + direction * FMM_STEP_SIZE * i
            qx, qy = int(round(q[0])), int(round(q[1]))
            if not (0 <= qx < w and 0 <= qy < h) or not binary_mask[qy, qx]:
                return out
            out.append(tuple(q))
        return []  # no wall within reach: keep the click as the end

    head = ray(pts)
    tail = ray(pts[::-1])
    if not head:
        print("  Start: no mask wall within reach, path starts at the click.")
    if not tail:
        print("  End: no mask wall within reach, path ends at the click.")
    return head[::-1] + list(map(tuple, pts)) + tail


def route_clicks(g, binary_mask, dist_map, points):
    """Route a skeleton-graph path through the clicked (x, y) ``points``.

    The first and last clicks become the path's own end nodes, joined to the
    skeleton by centre-seeking connectors (see ``_attach_endpoint``); intermediate
    waypoints snap to their nearest skeleton node. Returns ``(full_path, points, snap_px)`` where ``snap_px`` is each
    click's offset (px) from where the path actually passes it.
    """
    skeleton_nodes = set(g.nodes())
    g = g.copy()  # endpoint nodes are added temporarily
    node_coords = np.array(g.nodes())
    tree = KDTree(node_coords)

    path_nodes = []
    snap_px = []
    for i, pt in enumerate(points):
        if i == 0 or i == len(points) - 1:
            # The end click's edges are added only before the last segment, so it
            # can't serve as a shortcut for the segments before it.
            path_nodes.append(pt)
            snap_px.append(0.0)
            continue
        # points are (x, y), matching the (x, y) graph node coordinates
        dist, idx = tree.query(pt)
        path_nodes.append(tuple(node_coords[idx]))
        snap_px.append(float(dist))

    def attach(i):
        node, off = _attach_endpoint(g, tree, node_coords, binary_mask, dist_map, points[i])
        path_nodes[i], snap_px[i] = node, off
        return node

    start = attach(0)
    if len(points) == 1:
        return [start], points, snap_px

    full_path = [start]
    for i in range(len(path_nodes) - 1):
        if i == len(path_nodes) - 2:
            attach(i + 1)
        u = path_nodes[i]
        v = path_nodes[i + 1]
        try:
            # Find shortest path on graph between clicked points
            segment = nx.shortest_path(g, u, v, weight="weight")
        except nx.NetworkXNoPath:
            print(f"No path between {u} and {v}")
            return [], points, snap_px
        # Append segment (excluding start node to avoid duplication)
        full_path.extend(_expand_connectors(g, segment)[1:])
        if i == 0 and start not in skeleton_nodes:
            # Drop the start node so later segments can't shortcut through it.
            g.remove_node(start)

    return full_path, points, snap_px


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
    """Computes shortest path between two nodes using Fast Marching Method.

    Returns ``None`` when the gradient descent doesn't get back to ``start_node``
    (trapped in a local minimum, zero gradient, or left the grid).
    """
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
    # A path inside the mask can't be longer than ~2x its pixel count; beyond
    # that the descent is circling in a local minimum, not converging.
    max_steps = int(2 * np.count_nonzero(mask) / step_size) + 10

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
    else:
        return None

    if path[-1] != start_node:
        return None
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
        if segment is None:
            # Descent failed: keep the skeleton route for this stretch.
            print(f"  FMM segment {i} did not converge; using the skeleton path.")
            segment = [tuple(map(float, p)) for p in skeleton_segment]

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


@dataclass
class VeinProfile:
    """Per-point profiles along the centerline (what the QC page plots)."""

    path: np.ndarray  # (n, 2) centerline points in mask pixels, (x, y)
    s_mm: np.ndarray  # cumulative arc length at each point
    diameter_mm: np.ndarray  # smoothed lumen diameter
    signed_curvature: np.ndarray  # 1/mm, capped; sign = turn direction

    @property
    def curvature(self) -> np.ndarray:
        return np.abs(self.signed_curvature)


def calculate_metrics(
    longest_path: Sequence[Sequence[float]],
    binary_mask: np.ndarray,
    ps_x: float,
    ps_y: float,
) -> tuple[dict[str, float], dict[str, Any], VeinProfile]:
    """Computes the vein's metrics (length, tortuosity, diameter, curvature, turns,
    shape features) for the path.

    Returns ``(metrics, overlay, profile)``: ``metrics`` holds scalars only (one
    export row, see ``METRIC_COLUMNS``); ``overlay`` holds plot-only annotations
    (start/end nodes, sharpest-bend index, turn events); ``profile`` the per-point
    arrays.

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
    length_mm = float(cumulative_dist[-1])

    # --- COMPUTE METRICS ---

    # --- Tortuosity ---
    start_node = longest_path[0]
    end_node = longest_path[-1]
    chord_length_mm = np.sqrt(
        ((start_node[0] - end_node[0]) * ps_x) ** 2
        + ((start_node[1] - end_node[1]) * ps_y) ** 2
    )
    tortuosity = length_mm / chord_length_mm if chord_length_mm > 0 else 1.0

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

    # The path runs into the mask wall at both ends, where the distance transform
    # falls to ~0; summary stats skip DIAMETER_END_TRIM_DIAMETERS median diameters
    # at each end (as in shape_features) so the tips don't pose as the narrowest spot.
    trim = DIAMETER_END_TRIM_DIAMETERS * float(np.median(path_diameters_mm))
    core = (cumulative_dist >= trim) & (cumulative_dist <= length_mm - trim)
    if core.sum() < MIN_FEATURE_SAMPLES:
        core[:] = True
    core_d = path_diameters_mm[core]
    avg_diameter = np.mean(core_d)
    max_diameter = np.max(core_d)
    min_diameter = np.min(core_d)
    std_diameter = np.std(core_d)
    min_d_idx = int(np.flatnonzero(core)[np.argmin(core_d)])

    # --- Volume (Cylinders) ---
    segment_diameters = (path_diameters_mm[:-1] + path_diameters_mm[1:]) / 2
    segment_areas = np.pi * (segment_diameters / 2) ** 2
    volume_mm3 = np.sum(segment_areas * diffs_mm)

    # --- Curvature Profile (analytic, from a smoothing B-spline fit) ---
    # signed_curvature keeps the sign (turn direction) so curves can be counted by
    # inflection (sign change); curvature is its magnitude for the profile/stats.
    signed_curvature, _ = fit_centerline_spline(
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
    curv_capped_frac = float(np.mean(np.abs(signed_curvature) > kappa_max))
    signed_curvature = np.clip(signed_curvature, -kappa_max, kappa_max)

    curvature = np.abs(signed_curvature)

    if np.any(curvature):
        max_k_idx = int(np.argmax(curvature))
        max_k_val = float(curvature[max_k_idx])
        avg_curvature = float(np.mean(curvature))
        std_curvature = float(np.std(curvature))
    else:
        max_k_idx = 0
        max_k_val = 0.0
        avg_curvature = 0.0
        std_curvature = 0.0

    # Count curves from the spline curvature: bends between inflection points.
    turns_90, turns_180, turns_270, turn_events, inflections, total_curv = count_curves(
        signed_curvature, cumulative_dist
    )

    metrics = {
        "length": length_mm,
        "tortuosity": float(tortuosity),
        "avg_diameter": float(avg_diameter),
        "min_diameter": float(min_diameter),
        "max_diameter": float(max_diameter),
        "std_diameter": float(std_diameter),
        "volume": float(volume_mm3),
        "avg_curvature": avg_curvature,
        "max_curvature": max_k_val,
        "std_curvature": std_curvature,
        "inflection_count": inflections,
        "total_curvature": total_curv,
        "turns_90": turns_90,
        "turns_180": turns_180,
        "turns_270": turns_270,
        "curv_capped_frac": curv_capped_frac,
    }
    profile = VeinProfile(
        path=path_arr.astype(float),
        s_mm=cumulative_dist,
        diameter_mm=path_diameters_mm,
        signed_curvature=signed_curvature,
    )
    metrics.update(shape_features(profile, metrics))
    overlay = {
        "start_node": start_node,
        "end_node": end_node,
        "max_k_idx": max_k_idx,
        "min_d_idx": min_d_idx,
        "turn_events": turn_events,
    }
    return metrics, overlay, profile


def trimmed_diameter_samples(profile: VeinProfile) -> tuple[np.ndarray, np.ndarray]:
    """Diameter profile resampled to uniform arc length, ends trimmed.

    Returns ``(s_mm, diameter_mm)`` with ``DIAMETER_END_TRIM_DIAMETERS`` median
    diameters cut from each end, or the whole resampled profile when fewer than
    ``MIN_FEATURE_SAMPLES`` would remain. This is the stretch the diameter shape
    features are computed over (and the QC plot leaves unshaded).
    """
    s, d = profile.s_mm, profile.diameter_mm
    ds = np.diff(s)
    step = float(np.median(ds[ds > 0])) if np.any(ds > 0) else 0.0
    if step > 0:
        s_u = np.arange(s[0], s[-1] + step / 2, step)
        d_u = np.interp(s_u, s, d)
    else:
        s_u, d_u = s, d

    trim = DIAMETER_END_TRIM_DIAMETERS * float(np.median(d_u))
    keep = (s_u >= s_u[0] + trim) & (s_u <= s_u[-1] - trim)
    if keep.sum() >= MIN_FEATURE_SAMPLES:
        s_u, d_u = s_u[keep], d_u[keep]
    return s_u, d_u


def shape_features(profile: VeinProfile, metrics: dict[str, float]) -> dict[str, float]:
    """Scale-robust, length-normalized shape features for classification.

    Diameter features are taken over the profile resampled to uniform arc length
    (so every mm of vein weighs the same, whatever the point spacing) with
    ``DIAMETER_END_TRIM_DIAMETERS`` median diameters trimmed off each end, where the
    distance transform falls off toward the mask tips. Ratios to the median cancel
    any error in the estimated pixel spacing. Counts and the total turning come from
    ``metrics`` and are divided by the vein length in cm, so they don't just grow
    with how much of the vein was clicked.
    """
    s_u, d_u = trimmed_diameter_samples(profile)

    med = float(np.median(d_u))
    mean = float(np.mean(d_u))
    std = float(np.std(d_u))
    p10, p90 = (float(v) for v in np.percentile(d_u, [10, 90]))
    trimmed_cm = float(s_u[-1] - s_u[0]) / 10.0
    caliber_change = float(np.sum(np.abs(np.diff(d_u))))
    flat = std < 1e-9  # skew/kurtosis are undefined (NaN) on a constant profile

    length_cm = metrics["length"] / 10.0
    inv_cm = 1.0 / length_cm if length_cm > 0 else 0.0
    return {
        "diam_median": med,
        "diam_p10": p10,
        "diam_p90": p90,
        "diam_cv": std / mean if mean > 0 else 0.0,
        "ectasia_ratio": p90 / med if med > 0 else 0.0,
        "narrowing_ratio": p10 / med if med > 0 else 0.0,
        "dilated_frac": float(np.mean(d_u > DILATION_RATIO * med)),
        "diam_change_per_cm": (
            caliber_change / med / trimmed_cm if med > 0 and trimmed_cm > 0 else 0.0
        ),
        "diam_skew": 0.0 if flat else float(skew(d_u)),
        "diam_kurtosis": 0.0 if flat else float(kurtosis(d_u)),
        "turning_deg_per_cm": metrics["total_curvature"] * inv_cm,
        "inflections_per_cm": metrics["inflection_count"] * inv_cm,
        "turns90_per_cm": metrics["turns_90"] * inv_cm,
        "turns180_per_cm": metrics["turns_180"] * inv_cm,
        "icm": (metrics["inflection_count"] + 1) * metrics["tortuosity"],
    }


# Export columns: (metrics key, spreadsheet header, decimals in the xlsx/summary).
# The single source for column order/labels of metrics.xlsx, metrics.csv and the
# printed summary. Everything after the first block is length-normalized or
# dimensionless (see shape_features).
METRIC_COLUMNS = [
    ("length", "Length (mm)", 2),
    ("tortuosity", "Tortuosity", 2),
    ("avg_diameter", "Avg Diameter (mm)", 2),
    ("min_diameter", "Min Diameter (mm)", 2),
    ("max_diameter", "Max Diameter (mm)", 2),
    ("std_diameter", "Std Diameter (mm)", 2),
    ("volume", "Volume (mm³)", 2),
    ("avg_curvature", "Avg Curvature (1/mm)", 4),
    ("max_curvature", "Max Curvature (1/mm)", 4),
    ("std_curvature", "Std Curvature (1/mm)", 4),
    ("inflection_count", "Inflections", 0),
    ("total_curvature", "Total Turning (°)", 1),
    ("turns_90", "Turns ≥90°", 0),
    ("turns_180", "Turns ≥180°", 0),
    ("turns_270", "Turns ≥270°", 0),
    # Shape features
    ("diam_median", "Median Diameter (mm)", 3),
    ("diam_p10", "P10 Diameter (mm)", 3),
    ("diam_p90", "P90 Diameter (mm)", 3),
    ("diam_cv", "Diameter CV", 3),
    ("ectasia_ratio", "Ectasia Ratio (P90/median)", 3),
    ("narrowing_ratio", "Narrowing Ratio (P10/median)", 3),
    ("dilated_frac", f"Dilated Fraction (>{DILATION_RATIO:g}× median)", 3),
    ("diam_change_per_cm", "Caliber Change (1/cm)", 3),
    ("diam_skew", "Diameter Skewness", 3),
    ("diam_kurtosis", "Diameter Excess Kurtosis", 3),
    ("turning_deg_per_cm", "Turning (°/cm)", 1),
    ("inflections_per_cm", "Inflections (/cm)", 3),
    ("turns90_per_cm", "Turns ≥90° (/cm)", 3),
    ("turns180_per_cm", "Turns ≥180° (/cm)", 3),
    ("icm", "Inflection Count Metric", 2),
    ("curv_capped_frac", "Curvature Capped Fraction", 3),
]


def print_summary(metrics):
    """Prints a text summary of the calculated metrics."""
    width = max(len(header) for _, header, _ in METRIC_COLUMNS) + 2
    print("\n" + "=" * (width + 12))
    print("VEIN ANALYSIS SUMMARY".center(width + 12))
    print("=" * (width + 12))
    for key, header, decimals in METRIC_COLUMNS:
        print(f"{header + ':':<{width}}{metrics[key]:.{decimals}f}")
    print("=" * (width + 12) + "\n")


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


def _qc_status(metrics):
    return "REVIEW" if metrics.get("qc_flags") else "OK"


def export_all_metrics(all_metrics, out_dir):
    """Writes ``metrics.xlsx`` (for people) and ``metrics.csv`` (for models) to out_dir."""
    export_all_metrics_xlsx(all_metrics, os.path.join(out_dir, "metrics.xlsx"))
    export_all_metrics_csv(all_metrics, os.path.join(out_dir, "metrics.csv"))


def export_all_metrics_xlsx(all_metrics, output_path):
    """Exports metrics from all processed files to a single XLSX file (rounded)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Vein Metrics"

    ws.append(
        ["File", "Dataset"]
        + [header for _, header, _ in METRIC_COLUMNS]
        + ["QC Status", "QC Flags"]
    )
    for nii_path, metrics in all_metrics:
        ws.append(
            [os.path.basename(nii_path), _resolve_dataset(nii_path)[0].name]
            + [
                round(metrics[key], decimals) if decimals else metrics[key]
                for key, _, decimals in METRIC_COLUMNS
            ]
            + [_qc_status(metrics), " | ".join(metrics.get("qc_flags", []))]
        )

    # Auto-size columns (capped so the long QC-flags text doesn't make a huge column)
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)

    wb.save(output_path)
    print(f"Metrics saved to {output_path}")


def export_all_metrics_csv(all_metrics, output_path):
    """Exports the same rows as the XLSX to a CSV, unrounded, with snake_case keys
    as the header -- ready for ``pd.read_csv`` and a scikit-learn model."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["case", "dataset"]
            + [key for key, _, _ in METRIC_COLUMNS]
            + ["qc_status", "qc_flags"]
        )
        for nii_path, metrics in all_metrics:
            writer.writerow(
                [case_from_nii(nii_path), _resolve_dataset(nii_path)[0].name]
                + [metrics[key] for key, _, _ in METRIC_COLUMNS]
                + [_qc_status(metrics), " | ".join(metrics.get("qc_flags", []))]
            )
    print(f"Metrics saved to {output_path}")


def crop_bounds(binary_mask, pad=REPORT_PADDING):
    """``(x_min, x_max, y_min, y_max)`` of the mask plus padding, or None if empty."""
    y_indices, x_indices = np.where(binary_mask > 0)
    if len(y_indices) == 0:
        return None
    return (
        max(0, x_indices.min() - pad),
        min(binary_mask.shape[1], x_indices.max() + pad),
        max(0, y_indices.min() - pad),
        min(binary_mask.shape[0], y_indices.max() + pad),
    )


def draw_scale_bar(ax, ps_x, x_min, x_max, y_min, y_max, bar_mm=None):
    """Draw a calibrated scale bar (line with its mm length on top) in the lower right.

    ``bar_mm`` fixes the length (e.g. the image legend's own length); it is used
    only if it fits in ~60% of the view. Otherwise the largest "nice" value from
    SCALE_BAR_STEPS_MM fitting ~1/4 of the visible width is used. Converted to
    pixels with the horizontal spacing ``ps_x``.
    """
    view_w_mm = (x_max - x_min) * ps_x
    if bar_mm is None or bar_mm > 0.6 * view_w_mm:
        fitting = [s for s in SCALE_BAR_STEPS_MM if s <= view_w_mm / 4]
        bar_mm = fitting[-1] if fitting else SCALE_BAR_STEPS_MM[0]
    bar_px = bar_mm / ps_x

    margin_x = 0.05 * (x_max - x_min)
    margin_y = 0.07 * (y_max - y_min)
    x1 = x_max - margin_x
    x0 = x1 - bar_px
    y = y_max - margin_y
    outline = [pe.Stroke(linewidth=6, foreground="black"), pe.Normal()]
    ax.plot([x0, x1], [y, y], color="white", linewidth=3.5, solid_capstyle="butt",
            path_effects=outline, zorder=10)
    for xe in (x0, x1):  # end ticks
        ax.plot([xe, xe], [y - 0.012 * (y_max - y_min), y + 0.012 * (y_max - y_min)],
                color="white", linewidth=2, path_effects=outline, zorder=10)
    ax.text(
        (x0 + x1) / 2,
        y - 0.025 * (y_max - y_min),
        f"{bar_mm:g} mm",
        color="white",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="bottom",
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
        zorder=10,
    )
    ax.text(
        (x0 + x1) / 2,
        y + 0.02 * (y_max - y_min),
        f"{ps_x:.4f} mm/px",
        color="white",
        fontsize=8,
        ha="center",
        va="top",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        zorder=10,
    )


def draw_calibration_stripe(ax, p0, p1, value_mm, ps_x):
    """Overlay the measured calibration line exactly on the image's own scale bar.

    ``p0``/``p1`` are the (x, y) ends clicked on the legend when the spacing was
    measured; the line is drawn semi-transparent over them with the original
    physical length on top, so a misplaced click (spacing locked onto the wrong
    thing, or the wrong length typed) is obvious against the burned-in legend.
    """
    (x0, y0), (x1, y1) = p0, p1
    ax.plot([x0, x1], [y0, y1], color="red", linewidth=2, alpha=0.6,
            solid_capstyle="butt", zorder=10)
    # End ticks perpendicular to the stripe.
    length = np.hypot(x1 - x0, y1 - y0)
    if length > 0:
        nx_, ny_ = -(y1 - y0) / length, (x1 - x0) / length
        tick = max(3.0, 0.08 * length)
        for xe, ye in ((x0, y0), (x1, y1)):
            ax.plot([xe - nx_ * tick, xe + nx_ * tick], [ye - ny_ * tick, ye + ny_ * tick],
                    color="red", linewidth=2, alpha=0.8, zorder=10)
    ax.annotate(
        f"{value_mm:g} mm",
        ((x0 + x1) / 2, min(y0, y1)),
        xytext=(0, 12),
        textcoords="offset points",
        color="red",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="bottom",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        zorder=10,
    )
    ax.set_title(f"Scale calibration: {value_mm:g} mm = {length:.1f} px  "
                 f"({ps_x:.4f} mm/px)")


def arrow_indices(cumulative_dist, curvature, pts):
    """Path indices for direction arrows: evenly by arc length, denser in bends.

    Arrows sit at unit steps of ``s / base_spacing + turn / FLOW_ARROW_TURN_DEG``,
    where ``turn`` is the accumulated turning angle ∫|κ|ds in degrees, so a straight
    run gets ~FLOW_ARROW_COUNT arrows total and every tight turn adds more. Arrows
    closer than FLOW_ARROW_MIN_GAP_PX to the previous one are dropped.
    """
    total = cumulative_dist[-1]
    if total <= 0 or len(pts) < 3:
        return []
    ds = np.diff(cumulative_dist, prepend=0.0)
    turn_deg = np.degrees(np.cumsum(np.abs(curvature) * ds))
    u = cumulative_dist / (total / (FLOW_ARROW_COUNT + 1)) + turn_deg / FLOW_ARROW_TURN_DEG
    targets = np.arange(1, np.floor(u[-1]) + 1) - 0.5
    idxs, last = [], None
    for t in targets:
        i = int(np.searchsorted(u, t))
        if i <= 0 or i >= len(pts) - 1:
            continue
        if last is not None and np.hypot(*(pts[i] - pts[last])) < FLOW_ARROW_MIN_GAP_PX:
            continue
        idxs.append(i)
        last = i
    return idxs


def _halo(width=3, color="black"):
    """Path effect that outlines text/lines so they read on any background."""
    return [pe.withStroke(linewidth=width, foreground=color)]


def qc_checks(binary_mask, raw_img, calib, ps_x, metrics, snap_px):
    """Automatic sanity flags for the reviewer. Returns a list of messages (empty = OK).

    Each flag names something a human should look at on the QC page; none of them
    stop the analysis.
    """
    flags = []
    h, w = binary_mask.shape

    # Calibration
    if not calib:
        flags.append(
            "No spacing cache: spacing comes from the NIfTI header only and "
            "cannot be checked against the image legend."
        )
    else:
        if not calib.get("endpoints"):
            flags.append(
                "Scale-bar endpoints not saved: calibration cannot be shown on the "
                "legend. Re-measure with pipeline.py --manual-spacing."
            )
        native = calib.get("mm_per_px")
        lo, hi = PLAUSIBLE_MM_PER_PX
        if native is not None and not lo <= native <= hi:
            flags.append(f"Spacing {native:.4f} mm/px is outside the usual {lo}-{hi} range.")
        if native is not None and raw_img is not None:
            header_native = ps_x * w / raw_img.shape[1]  # undo the upscale
            if abs(header_native - native) / native > QC_SPACING_MISMATCH_FRAC:
                flags.append(
                    f"NIfTI spacing ({header_native:.4f} mm/px native) disagrees with "
                    f"the spacing cache ({native:.4f} mm/px)."
                )
    if raw_img is None:
        flags.append("Source image not found: segmentation cannot be checked against it.")

    # Segmentation
    labels, _ = ndimage_label(binary_mask)
    sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1]
    sizes = sizes[sizes >= QC_MIN_PIECE_PX]
    if len(sizes) > 1:
        flags.append(
            f"Mask has {len(sizes)} separate pieces (main {sizes[0]} px, next "
            f"{sizes[1]} px): stray blobs or a broken vessel?"
        )
    if binary_mask[0].any() or binary_mask[-1].any() or binary_mask[:, 0].any() \
            or binary_mask[:, -1].any():
        flags.append("Mask touches the image border: vessel may be cut off.")

    # Path
    far = [i + 1 for i, d in enumerate(snap_px) if d * ps_x > QC_CLICK_SNAP_WARN_MM]
    if far:
        flags.append(
            f"Click(s) {', '.join(map(str, far))} are > {QC_CLICK_SNAP_WARN_MM:g} mm "
            "from the path: path may not go where intended."
        )

    # Measurements
    min_px = metrics["min_diameter"] / ps_x
    if min_px < QC_MIN_DIAMETER_PX:
        flags.append(
            f"Min diameter is only {min_px:.1f} px wide: at the resolution limit, "
            "treat it as approximate."
        )
    capped = metrics.get("curv_capped_frac", 0.0)
    if capped > QC_CURV_CAPPED_WARN_FRAC:
        flags.append(
            f"{capped:.0%} of curvature samples hit the physical cap: likely "
            "projection folds, check the turns."
        )
    return flags


def _crop_axes(ax, bounds):
    x_min, x_max, y_min, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)


def _show_background(ax, img, extent, shape):
    if img is not None:
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, extent=extent)
    else:
        ax.imshow(np.zeros(shape), cmap="gray", extent=extent)


def _colored_path(ax, xy, values, cmap, norm, width=3):
    """Draw a polyline colored by ``values`` (one per vertex) with a dark outline."""
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=width,
                        path_effects=[pe.Stroke(linewidth=width + 2, foreground="black"),
                                      pe.Normal()])
    lc.set_array((values[:-1] + values[1:]) / 2)
    ax.add_collection(lc)
    return lc


def _local_directions(pts, half_window=5):
    """Unit tangent (x, y) at each path point from a centred chord."""
    idx = np.arange(len(pts))
    a = pts[np.clip(idx + half_window, 0, len(pts) - 1)]
    b = pts[np.clip(idx - half_window, 0, len(pts) - 1)]
    d = a - b
    n = np.hypot(d[:, 0], d[:, 1])
    n[n == 0] = 1
    return d / n[:, None]


def visualize_qc(
    binary_mask,
    ps_x,
    ps_y,
    profile,
    metrics,
    overlay,
    clicks,
    snap_px,
    filename,
):
    """Save the one-page QC report ``{case}_qc.png`` and return its flag list.

    Laid out as the questions a reviewer works through:
      1. calibration   - red line on the image's own scale bar, with its mm
      2. segmentation  - raw image (and the clinician's tracing, if any) next to it
      3. path          - mask outline, centerline colored start->end, direction
                         arrows (denser in bends), numbered clicks
      4. measurements  - diameter drawn as width lines across the vessel, min
                         diameter and numbered turns marked on the image
      5. profiles      - diameter/curvature vs distance, colored like the path
      6. summary       - metrics and the automatic flags (qc_checks)
    The flags are also stored in the PNG metadata for the dataset PDF's cover page.
    """
    raw_img, tracing_img = get_reference_images(filename)
    calib = load_scale_calibration(filename) or {}
    flags = qc_checks(binary_mask, raw_img, calib, ps_x, metrics, snap_px)

    h, w = binary_mask.shape
    extent = (-0.5, w - 0.5, h - 0.5, -0.5)  # stretch native images onto the mask grid
    bounds = crop_bounds(binary_mask) or (0, w, 0, h)
    pts = profile.path
    cumulative_dist = profile.s_mm
    path_diameters_mm = profile.diameter_mm
    curvature = profile.curvature
    max_k_idx = overlay["max_k_idx"]
    turn_events = overlay["turn_events"]
    total = cumulative_dist[-1] if cumulative_dist[-1] > 0 else 1.0
    cmap = plt.get_cmap(FLOW_CMAP)
    norm = plt.Normalize(0, total)

    stripe = None
    if raw_img is not None and calib.get("endpoints") and calib.get("value_mm"):
        sy, sx = h / raw_img.shape[0], w / raw_img.shape[1]
        stripe = [(c * sx, r * sy) for r, c in calib["endpoints"][:2]]

    # Size rows from the crops' aspect ratios so tall/wide veins don't leave gaps.
    fig_w = 22.0
    top = [p for p in ("cal" if stripe else None, "raw",
                       "trace" if tracing_img is not None else None) if p]
    vein_aspect = (bounds[3] - bounds[2]) / max(bounds[1] - bounds[0], 1)
    cal_box = None
    if stripe:
        (x0, y0), (x1, y1) = stripe
        pad = max(20.0, 0.8 * np.hypot(x1 - x0, y1 - y0))
        cal_box = (max(-0.5, min(x0, x1) - pad), min(w - 0.5, max(x0, x1) + pad),
                   max(-0.5, min(y0, y1) - pad), min(h - 0.5, max(y0, y1) + pad))
    top_w = 0.85 * fig_w / len(top)
    h1 = max(min(top_w * (
        (cal_box[3] - cal_box[2]) / max(cal_box[1] - cal_box[0], 1) if k == "cal"
        else vein_aspect), 9.0) for k in top) + 0.8
    h2 = min(0.42 * fig_w * vein_aspect, 13.0) + 1.6  # + room for the colorbar
    h3 = 6.0
    fig_h = h1 + h2 + h3 + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    status = "OK" if not flags else f"REVIEW ({len(flags)} flag{'s' * (len(flags) > 1)})"
    fig.suptitle(
        f"Vein QC: {case_from_nii(filename)}   —   {status}",
        fontsize=20,
        fontweight="bold",
        color="tab:green" if not flags else "tab:orange",
    )
    outer = fig.add_gridspec(3, 1, height_ratios=[h1, h2, h3], hspace=0.12,
                             top=1 - 0.9 / fig_h, bottom=0.3 / fig_h)

    # --- Row 1: calibration | raw | tracing -------------------------------------
    row1 = outer[0].subgridspec(1, len(top), wspace=0.12)
    for k, kind in enumerate(top):
        ax = fig.add_subplot(row1[0, k])
        if kind == "cal":
            _show_background(ax, raw_img, extent, (h, w))
            draw_calibration_stripe(ax, stripe[0], stripe[1], calib["value_mm"], ps_x)
            _crop_axes(ax, cal_box)
            ax.set_title(
                f"1 · Calibration — red line on the legend bar?\n"
                f"{calib['value_mm']:g} mm = {np.hypot(x1 - x0, y1 - y0):.1f} px "
                f"({ps_x:.4f} mm/px)",
                fontsize=12,
            )
        elif kind == "raw":
            _show_background(ax, raw_img, extent, (h, w))
            _crop_axes(ax, bounds)
            draw_scale_bar(ax, ps_x, *bounds, calib.get("value_mm"))
            ax.set_title("2 · Raw image (unannotated)", fontsize=12)
        else:
            _show_background(ax, tracing_img, extent, (h, w))
            _crop_axes(ax, bounds)
            ax.set_title("2 · Clinician tracing — does the mask match?", fontsize=12)

    # --- Row 2: path | measurements ---------------------------------------------
    row2 = outer[1].subgridspec(1, 2, wspace=0.08)
    ax_path = fig.add_subplot(row2[0, 0])
    ax_meas = fig.add_subplot(row2[0, 1])
    fill = np.zeros((h, w, 4), dtype=np.float32)
    fill[binary_mask > 0] = (0.0, 0.8, 1.0, 0.18)
    for ax in (ax_path, ax_meas):
        _show_background(ax, raw_img, extent, (h, w))
        ax.imshow(fill, extent=extent)
        ax.contour(binary_mask, levels=[0.5], colors="cyan", linewidths=1.0)
        _crop_axes(ax, bounds)
        draw_scale_bar(ax, ps_x, *bounds, calib.get("value_mm"))

    # Path panel: colored centerline, arrows, clicks, start/end.
    lc = _colored_path(ax_path, pts, cumulative_dist, cmap, norm)
    for i in arrow_indices(cumulative_dist, curvature, pts):
        i0, i1 = max(0, i - 3), min(len(pts) - 1, i + 3)
        if np.hypot(*(pts[i1] - pts[i0])) < 1e-6:
            continue
        ax_path.annotate(
            "", xy=pts[i1], xytext=pts[i0], zorder=7,
            arrowprops=dict(arrowstyle="-|>", color=cmap(norm(cumulative_dist[i])),
                            lw=2, mutation_scale=22, shrinkA=0, shrinkB=0,
                            path_effects=_halo(4)),
        )
    for k, (cx, cy) in enumerate(clicks, start=1):
        far = snap_px and snap_px[k - 1] * ps_x > QC_CLICK_SNAP_WARN_MM
        ax_path.scatter(cx, cy, s=160, c="red" if far else "white",
                        edgecolors="black", zorder=9)
        ax_path.text(cx, cy, str(k), fontsize=8, fontweight="bold", ha="center",
                     va="center", zorder=10)
    for node, label, color in (
        (overlay["start_node"], "START", "lime"),
        (overlay["end_node"], "END", "magenta"),
    ):
        ax_path.annotate(label, node, xytext=(10, 10), textcoords="offset points",
                         color=color, fontsize=12, fontweight="bold", zorder=10,
                         path_effects=_halo())
    cbar = fig.colorbar(lc, ax=ax_path, orientation="horizontal", fraction=0.04,
                        pad=0.06, aspect=40)
    cbar.set_label("Distance along vein (mm)")
    ax_path.set_title(
        "3 · Mask + path — right vessel, no leaks/gaps, clicks (numbered) on it?",
        fontsize=12,
    )

    # Measurements panel: width lines, min diameter, numbered turns.
    ax_meas.plot(pts[:, 0], pts[:, 1], color="white", lw=1, alpha=0.7, zorder=5)
    tangents = _local_directions(pts)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    px_len_mm = np.hypot(normals[:, 0] * ps_x, normals[:, 1] * ps_y)  # mm per px along n
    interval = max(QC_WIDTH_INTERVAL_MM, cumulative_dist[-1] / QC_MAX_WIDTH_LINES)
    width_idx = np.searchsorted(cumulative_dist, np.arange(interval / 2, cumulative_dist[-1],
                                                           interval))
    width_idx = width_idx[width_idx < len(pts)]
    min_idx = overlay["min_d_idx"]

    def width_line(i, **kw):
        half = path_diameters_mm[i] / 2 / px_len_mm[i]
        a, b = pts[i] - normals[i] * half, pts[i] + normals[i] * half
        ax_meas.plot([a[0], b[0]], [a[1], b[1]], solid_capstyle="butt", **kw)

    for i in width_idx:
        width_line(i, color="white", lw=1.5, zorder=7, path_effects=_halo(3))
    width_line(min_idx, color="red", lw=3, zorder=8, path_effects=_halo(5))
    ax_meas.annotate(f"min Ø {path_diameters_mm[min_idx]:.2f} mm", pts[min_idx],
                     xytext=(12, -16), textcoords="offset points", color="red",
                     fontsize=11, fontweight="bold", zorder=10,
                     path_effects=_halo(3, "white"))

    for k, (path_idx, angle_deg) in enumerate(turn_events, start=1):
        if path_idx >= len(pts):
            continue
        tx, ty = pts[path_idx]
        color = turn_color(angle_deg)
        ax_meas.scatter(tx, ty, facecolors="none", edgecolors=color, s=260,
                        linewidths=2.5, zorder=8)
        # Alternate above/below so labels of clustered turns don't overprint.
        ax_meas.annotate(f"T{k} {angle_deg:.0f}°", (tx, ty),
                         xytext=(10, 8) if k % 2 else (10, -20),
                         textcoords="offset points", color=color, fontsize=10,
                         fontweight="bold", zorder=10, path_effects=_halo(3, "white"))
    mk = pts[max_k_idx]
    ax_meas.scatter(mk[0], mk[1], c="yellow", marker="x", s=150, linewidth=3, zorder=9)
    ax_meas.set_title(
        "4 · Measurements — width lines span the vessel? turns (T#) real bends?",
        fontsize=12,
    )

    # --- Row 3: profiles | summary ----------------------------------------------
    row3 = outer[2].subgridspec(1, 3, wspace=0.25, width_ratios=[1, 1, 0.9])
    ax_d = fig.add_subplot(row3[0, 0])
    ax_k = fig.add_subplot(row3[0, 1])
    ax_s = fig.add_subplot(row3[0, 2])

    for ax, values, ylabel in (
        (ax_d, path_diameters_mm, "Diameter (mm)"),
        (ax_k, curvature, "Curvature (1/mm)"),
    ):
        xy = np.column_stack([cumulative_dist, values])
        _colored_path(ax, xy, cumulative_dist, cmap, norm, width=2.5)
        ax.fill_between(cumulative_dist, values, color="gray", alpha=0.1)
        ax.set_xlim(0, cumulative_dist[-1])
        ax.set_ylim(0, max(float(np.max(values)) * 1.15, 1e-6))
        ax.set_xlabel("Distance along vein (mm)  —  colors match the path")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Grey out the ends trimmed before the diameter shape features are computed.
    s_kept, _ = trimmed_diameter_samples(profile)
    excluded = [(cumulative_dist[0], s_kept[0]), (s_kept[-1], cumulative_dist[-1])]
    label = "Excluded from shape features"
    for lo, hi in excluded:
        if hi > lo:
            ax_d.axvspan(lo, hi, color="gray", alpha=0.3, lw=0, zorder=0, label=label)
            label = None
    ax_d.axhline(metrics["avg_diameter"], color="black", ls="--", alpha=0.5,
                 label=f"Mean {metrics['avg_diameter']:.2f} mm")
    ax_d.scatter(cumulative_dist[min_idx], path_diameters_mm[min_idx], c="red",
                 edgecolors="black", s=60, zorder=5, label="Min (red line on image)")
    ax_d.legend(loc="upper right", fontsize=9)
    ax_d.set_title("5 · Diameter profile", fontsize=12)

    for k, (path_idx, angle_deg) in enumerate(turn_events, start=1):
        if path_idx >= len(cumulative_dist):
            continue
        color = turn_color(angle_deg)
        ax_k.axvline(cumulative_dist[path_idx], color=color, ls=":", lw=1.5, alpha=0.8)
        ax_k.text(cumulative_dist[path_idx], 0.98, f"T{k} {angle_deg:.0f}°",
                  transform=ax_k.get_xaxis_transform(), rotation=90, va="top",
                  ha="right", fontsize=9, color=color)
    ax_k.scatter(cumulative_dist[max_k_idx], curvature[max_k_idx], c="yellow",
                 marker="x", s=100, linewidth=2.5, zorder=5, label="Max bend")
    ax_k.legend(loc="upper left", fontsize=9)
    ax_k.set_title("5 · Curvature profile (turns numbered as on the image)", fontsize=12)

    ax_s.axis("off")
    summary = (
        f"Length        {metrics['length']:.2f} mm\n"
        f"Tortuosity    {metrics['tortuosity']:.2f}\n"
        f"Diameter      {metrics['avg_diameter']:.2f} mm avg\n"
        f"              {metrics['min_diameter']:.2f}-{metrics['max_diameter']:.2f} mm "
        f"(sd {metrics['std_diameter']:.2f})\n"
        f"Volume        {metrics['volume']:.2f} mm³\n"
        f"Curvature     {metrics['avg_curvature']:.3f} avg / "
        f"{metrics['max_curvature']:.3f} max mm⁻¹\n"
        f"Inflections   {metrics['inflection_count']}   total turn "
        f"{metrics['total_curvature']:.0f}° ({metrics['turning_deg_per_cm']:.0f}°/cm)\n"
        f"Turns         {metrics['turns_90']} ≥90°  {metrics['turns_180']} ≥180°  "
        f"{metrics['turns_270']} ≥270°\n"
        f"Shape         CV {metrics['diam_cv']:.2f}  ectasia "
        f"{metrics['ectasia_ratio']:.2f}  narrowing {metrics['narrowing_ratio']:.2f}\n"
        f"Spacing       {ps_x:.4f} × {ps_y:.4f} mm/px"
    )
    ax_s.text(0, 1, "6 · Summary", fontsize=13, fontweight="bold", va="top")
    ax_s.text(0, 0.9, summary, family="monospace", fontsize=10.5, va="top")
    flag_text = (
        "\n".join("• " + textwrap.fill(f, 52, subsequent_indent="  ") for f in flags)
        if flags else "No automatic flags. Still check panels 1-5."
    )
    ax_s.text(0, 0.36, "Automatic checks", fontsize=12, fontweight="bold", va="top",
              color="tab:green" if not flags else "tab:orange")
    ax_s.text(0, 0.29, flag_text, fontsize=9.5, va="top", wrap=True)

    fig.savefig(
        get_report_path(filename),
        dpi=QC_DPI,
        bbox_inches="tight",
        metadata={"Description": json.dumps({"status": "OK" if not flags else "REVIEW",
                                             "flags": flags})},
    )
    plt.close(fig)
    return flags


def natural_key(path):
    """Sort key so 1m, 2m, ... 10m order naturally (and JAP/TJU group together)."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def export_qc_pdf(data_dir, output_path=None):
    """Bundle every ``{case}_qc.png`` in ``data_dir`` into one PDF for review.

    The first page lists every case with its status and flags (read back from each
    PNG's metadata), flagged cases first, so a reviewer knows where to look.
    """
    from PIL import Image

    output_path = output_path or os.path.join(data_dir, "qc_report.pdf")
    pngs = sorted(glob.glob(os.path.join(data_dir, "*" + QC_SUFFIX)), key=natural_key)
    if not pngs:
        print("No QC pages to bundle.")
        return

    rows = []
    for p in pngs:
        with Image.open(p) as im:
            info = json.loads(im.info.get("Description", "{}") or "{}")
        rows.append((os.path.basename(p)[: -len(QC_SUFFIX)], info.get("status", "?"),
                     info.get("flags", [])))
    n_review = sum(1 for _, s, _ in rows if s != "OK")

    # Cover page: flagged cases first, then OK ones, each with page number.
    per_page = 42
    n_cover = -(-len(rows) // per_page)
    order = sorted(range(len(rows)), key=lambda i: (rows[i][1] == "OK", i))
    lines = []
    for i in order:
        case, status, flags = rows[i]
        lines.append((f"p.{n_cover + i + 1:<3} {case:<10} {status}", flags))
    cover_pages = []
    for start in range(0, len(lines), per_page):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.96, "Vein QC report", fontsize=18, fontweight="bold")
        fig.text(0.06, 0.935, f"{len(rows)} cases · {n_review} need review · "
                 f"{len(rows) - n_review} OK", fontsize=11)
        y = 0.9
        for head, flags in lines[start:start + per_page]:
            color = "tab:green" if not flags else "tab:orange"
            fig.text(0.06, y, head, family="monospace", fontsize=9, color=color,
                     fontweight="bold")
            if flags:
                short = "; ".join(f.split(":")[0] for f in flags)
                fig.text(0.34, y, textwrap.shorten(short, 95), fontsize=7.5)
            y -= 0.02
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        cover_pages.append(Image.fromarray(buf.copy()))
        plt.close(fig)

    pages = [Image.open(p).convert("RGB") for p in pngs]
    first, *rest = cover_pages + pages
    first.save(output_path, save_all=True, append_images=rest, resolution=QC_DPI)
    for im in pages:
        im.close()
    print(f"QC report ({len(pngs)} cases, {n_review} flagged) -> {output_path}")


def analyze_vein(nii_path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Main function to orchestrate the vein analysis pipeline."""
    binary_mask, ps_x, ps_y = load_and_preprocess(nii_path, MANUAL_PIXEL_SIZE)
    g, dist_map = skeletonize_and_graph(binary_mask, ps_x, ps_y)

    # Get path from user
    longest_path_graph, clicks, snap_px = get_user_path(g, binary_mask, dist_map, nii_path)

    if not longest_path_graph:
        print("No path found.")
        return None

    # Refine path with FMM (follows loops)
    longest_path = compute_centerline_fmm(
        binary_mask, longest_path_graph, dist_map, ps_x, ps_y
    )
    # Bind the ends to the segmentation: run on past the clicks to the mask wall.
    longest_path = extend_to_wall(longest_path, binary_mask, dist_map)

    metrics, overlay, profile = calculate_metrics(longest_path, binary_mask, ps_x, ps_y)

    print_summary(metrics)
    flags = visualize_qc(
        binary_mask, ps_x, ps_y, profile, metrics, overlay, clicks, snap_px, nii_path
    )
    metrics["qc_flags"] = flags
    for f in flags:
        print(f"  QC flag: {f}")

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
        export_all_metrics(all_metrics, data_dir)
        export_qc_pdf(data_dir)
