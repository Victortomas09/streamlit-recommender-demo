import numpy as np
import plotly.graph_objects as go  


def pad_truncate(vec, k, fill_val=0.0):
    """
    Pad or truncate a vector to length k.
    """
    vec = np.array(vec)
    if vec.shape[0] < k:
        vec = np.hstack([vec, np.full(k - vec.shape[0], fill_val)])
    elif vec.shape[0] > k:
        vec = vec[:k]
    return vec

# ------------------------
# Helper functions
# ------------------------
def pad_columns_2d(mat, target_cols, fill_val=0.0):
    """Pad or truncate the columns of a 2D matrix to target_cols, keep all rows."""
    n_rows, n_cols = mat.shape
    if n_cols < target_cols:
        mat_padded = np.hstack([mat, np.full((n_rows, target_cols - n_cols), fill_val)])
    else:
        mat_padded = mat[:, :target_cols]
    return mat_padded


def prepare_user_trajectory(session_traj, k, fill_val_vec, 
                            exaggeration=1.0, drift_target=None, drift_strength=0.0, center_offset=None):
    """
    Pads/truncates a trajectory, optionally exaggerates, drifts toward target, 
    and centers relative to an offset.
    """
    traj = prepare_trajectory(session_traj, k, fill_val_vec)[:, :3]  # keep first 3 dims
    start = traj[0]
    
    # exaggerate
    traj_offsets = traj - start
    traj = start + traj_offsets * exaggeration
    
    # drift toward target
    if drift_target is not None and drift_strength > 0:
        direction = drift_target - traj[-1]
        traj += np.linspace(0, 1, len(traj)).reshape(-1,1) * direction * drift_strength

    # center
    if center_offset is not None:
        traj = center_vectors(traj, center_offset)
        
    return traj

def plot_top10_markers(fig, df, color_map, symbol="diamond", size=9, name="Top-10"):
    """Add top-10 markers to a plotly 3D figure."""
    if not df.empty:
        fig.add_trace(go.Scatter3d(
            x=df["dim1"], y=df["dim2"], z=df["dim3"],
            mode="markers+text",
            name=name,
            marker=dict(size=size, symbol=symbol, opacity=1,
                        color=[color_map.get(c, "gray") for c in df["category"]]),
            text=df["title"], textposition="top center"
        ))

def plot_category_scatter(fig, df, category_colors, size=5, name_suffix="Movies"):
    """Plot movies by category with color mapping."""
    for cat, color in category_colors.items():
        subset = df[df["category"] == cat]
        if not subset.empty:
            fig.add_trace(go.Scatter3d(
                x=subset["dim1"], y=subset["dim2"], z=subset["dim3"],
                mode="markers",
                name=f"{cat.capitalize()} {name_suffix}",
                marker=dict(size=size, color=color, opacity=0.6),
                text=subset["title"]
            ))


def center_vectors(arr, center_vec):
    """
    Center rows of arr by center_vec.
    Handles 1D or 2D arrays.
    """
    arr = np.array(arr)
    center_vec = np.array(center_vec)
    if arr.ndim == 1:
        return arr - center_vec
    return arr - center_vec.reshape(1, -1)

def normalize_rows(arr, eps=1e-9):
    """
    Normalize rows to unit length.
    """
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + eps)

def prepare_trajectory(traj_list, k, fill_val_vec=None):
    """
    Convert list of vectors to np.array, pad/truncate to k, optionally fill missing dims with fill_val_vec
    """
    traj_arr = np.array(traj_list)
    if traj_arr.ndim == 1:
        traj_arr = traj_arr.reshape(1, -1)
    n_steps, dim = traj_arr.shape
    if dim < k:
        if fill_val_vec is None:
            fill_vals = np.zeros((n_steps, k - dim))
        else:
            fill_vals = np.tile(fill_val_vec, (n_steps, 1))[:, : (k - dim)]
        traj_arr = np.hstack([traj_arr, fill_vals])
    elif dim > k:
        traj_arr = traj_arr[:, :k]
    return traj_arr


# ------------------------
# Helper functions
# ------------------------
def pad_truncate_2d(arr2d, target_cols, fill_val=0.0):
    """Pad or truncate 2D array to target number of columns."""
    n_rows, n_cols = arr2d.shape
    if n_cols < target_cols:
        pad_width = target_cols - n_cols
        return np.hstack([arr2d, np.full((n_rows, pad_width), fill_val)])
    else:
        return arr2d[:, :target_cols]

def prepare_user_trajectory(session_traj, k, fill_val_vec, 
                            exaggeration=1.0, drift_target=None, drift_strength=0.0, center_offset=None):
    """
    Pads/truncates a trajectory, optionally exaggerates, drifts toward target, 
    and centers relative to an offset.
    """
    traj = prepare_trajectory(session_traj, k, fill_val_vec)[:, :3]  # keep first 3 dims
    start = traj[0]
    
    # exaggerate
    traj_offsets = traj - start
    traj = start + traj_offsets * exaggeration
    
    # drift toward target
    if drift_target is not None and drift_strength > 0:
        direction = drift_target - traj[-1]
        traj += np.linspace(0, 1, len(traj)).reshape(-1,1) * direction * drift_strength

    # center
    if center_offset is not None:
        traj = center_vectors(traj, center_offset)
        
    return traj

def plot_top10_markers(fig, df, color_map, symbol="diamond", size=9, name="Top-10"):
    """Add top-10 markers to a plotly 3D figure."""
    if not df.empty:
        fig.add_trace(go.Scatter3d(
            x=df["dim1"], y=df["dim2"], z=df["dim3"],
            mode="markers+text",
            name=name,
            marker=dict(size=size, symbol=symbol, opacity=1,
                        color=[color_map.get(c, "gray") for c in df["category"]]),
            text=df["title"], textposition="top center"
        ))

def plot_category_scatter(fig, df, category_colors, size=5, name_suffix="Movies"):
    """Plot movies by category with color mapping."""
    for cat, color in category_colors.items():
        subset = df[df["category"] == cat]
        if not subset.empty:
            fig.add_trace(go.Scatter3d(
                x=subset["dim1"], y=subset["dim2"], z=subset["dim3"],
                mode="markers",
                name=f"{cat.capitalize()} {name_suffix}",
                marker=dict(size=size, color=color, opacity=0.6),
                text=subset["title"]
            ))



