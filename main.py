import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import skfmm
import os
import glob
from skimage.morphology import medial_axis
from skimage.draw import disk
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.spatial import KDTree

# ==========================================
# CONFIGURATION
# ==========================================
MANUAL_PIXEL_SIZE = None  # Set to e.g., (0.5, 0.5) for manual pixel size in mm (x, y)

# Analysis parameters
FMM_WAYPOINT_INTERVAL_MM = 2.0  # Interval for waypoint sampling in FMM refinement
DIAMETER_SMOOTHING_SIGMA = 3.0  # Gaussian smoothing sigma for diameter profile
CURVATURE_SMOOTHING_SIGMA = 5.0  # Gaussian smoothing sigma for curvature calculation
CORRIDOR_RADIUS_FACTOR = 1.2  # Factor to expand corridor around skeleton for FMM
MIN_CORRIDOR_RADIUS = 1.5  # Minimum corridor radius in pixels
FMM_STEP_SIZE = 0.5  # Step size for FMM gradient descent
VIEW_PADDING = 20  # Padding around segmentation for interactive view
REPORT_PADDING = 40  # Padding around segmentation for report visualization


def get_report_path(nii_path):
    """Returns the report path for a given NIfTI file, handling .nii and .nii.gz."""
    if nii_path.endswith(".nii.gz"):
        return nii_path[:-7] + "_report.png"
    elif nii_path.endswith(".nii"):
        return nii_path[:-4] + "_report.png"
    else:
        return nii_path + "_report.png"


def load_and_preprocess(nii_path, manual_pixel_size=None):
    """Loads NIfTI file and extracts binary mask and pixel dimensions."""
    print(f"Loading {nii_path}...")
    img = nib.load(nii_path)
    data = np.asarray(img.dataobj)
    data_2d = np.squeeze(data)
    binary_mask = (data_2d > 0).astype(np.uint8)

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


def get_user_path(g, binary_mask):
    """Allows user to manually select path points on an interactive plot."""
    print("Please select the path manually on the plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(binary_mask, cmap="gray")

    # Show graph edges
    for u, v in g.edges():
        ax.plot([u[0], v[0]], [u[1], v[1]], "c-", linewidth=0.5, alpha=0.5)

    ax.set_title(
        "LEFT CLICK to add points in order.\nRIGHT CLICK to remove last point.\nENTER to finish."
    )

    # Crop to segmentation
    y_indices, x_indices = np.where(binary_mask > 0)
    if len(y_indices) > 0 and len(x_indices) > 0:
        pad = VIEW_PADDING
        y_min = max(0, y_indices.min() - pad)
        y_max = min(binary_mask.shape[0], y_indices.max() + pad)
        x_min = max(0, x_indices.min() - pad)
        x_max = min(binary_mask.shape[1], x_indices.max() + pad)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

    print("Interactive plot opened. Please click points in the window.")

    # Get user clicks (unlimited until Enter)
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    if not points:
        print("No points selected.")
        return []

    print(f"User selected {len(points)} points. Calculating path...")

    # Map clicks to nearest graph nodes
    node_coords = np.array(g.nodes())
    tree = KDTree(node_coords)

    path_nodes = []
    for pt in points:
        # ginput returns (x, y)
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


def calculate_metrics(
    longest_path, binary_mask, ps_x, ps_y, max_length_mm, dist_map=None
):
    """Computes geometric metrics (length, tortuosity, diameter, curvature) for the path."""
    # Convert to numpy for calculations
    path_arr = np.array(longest_path)
    path_x_mm = path_arr[:, 0] * ps_x
    path_y_mm = path_arr[:, 1] * ps_y

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
    if dist_map is None:
        dist_map = distance_transform_edt(binary_mask)

    path_radii = np.array(
        [dist_map[int(round(p[1])), int(round(p[0]))] for p in longest_path]
    )
    path_diameters_mm = path_radii * 2 * ps_x

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

    # --- Curvature Profile ---
    # Smooth path to reduce derivative noise
    sigma = CURVATURE_SMOOTHING_SIGMA
    path_x_smooth = gaussian_filter1d(path_x_mm, sigma=sigma)
    path_y_smooth = gaussian_filter1d(path_y_mm, sigma=sigma)

    if len(path_x_mm) > 3:
        dx = np.gradient(path_x_smooth)
        dy = np.gradient(path_y_smooth)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        num = np.abs(dx * ddy - dy * ddx)
        den = np.power(dx**2 + dy**2, 1.5)
        curvature = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

        max_k_idx = np.argmax(curvature)
        max_k_point = longest_path[max_k_idx]
        max_k_val = curvature[max_k_idx]
        avg_curvature = np.mean(curvature)
        std_curvature = np.std(curvature)
    else:
        curvature = np.zeros(len(path_x_mm))
        avg_curvature = 0
        std_curvature = 0
        max_k_point = longest_path[0]
        max_k_val = 0
        max_k_idx = 0

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
    }

    return metrics, cumulative_dist, path_diameters_mm, curvature, dist_map


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
    print("=" * 30 + "\n")


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
        path_x_plot, path_y_plot, "r-", linewidth=2.0, alpha=0.8, label="Centerline"
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
        f"Std: {metrics['std_curvature']:.4f} mm⁻¹"
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
    longest_path_graph = get_user_path(g, binary_mask)

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

    metrics, cumulative_dist, path_diameters_mm, curvature, dist_map = (
        calculate_metrics(
            longest_path, binary_mask, ps_x, ps_y, max_length_mm, dist_map
        )
    )

    print_summary(metrics)
    visualize_results(
        binary_mask,
        dist_map,
        longest_path,
        cumulative_dist,
        path_diameters_mm,
        curvature,
        metrics,
        nii_path,
    )


if __name__ == "__main__":
    data_dir = "data"
    nii_files = glob.glob(os.path.join(data_dir, "*.nii")) + glob.glob(
        os.path.join(data_dir, "*.nii.gz")
    )

    for nii_path in nii_files:
        report_path = get_report_path(nii_path)
        if os.path.exists(report_path):
            print(f"Skipping {nii_path} (Report already exists)")
            continue

        print(f"Processing {nii_path}...")
        analyze_vein(nii_path)
