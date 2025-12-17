import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objects as go  
import plotly.express as px
from umap import UMAP
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from helpers.helpers_tab1 import get_category_mapping_df, plot_category_distribution, plot_user_political_extremeness

from helpers.helpers_tab2 import (
    build_rating_matrices,
    run_svd, run_als, run_pmf,
    plot_top10_category_bar,
    plot_latent_3d,
)

from helpers.helpers_latex_svd import show_svd_block
from helpers.helpers_latex_pmf import show_pmf_block
from helpers.helpers_latex_als import show_als_block
from helpers.helpers_latex_mse import show_error_metrics
from helpers.helpers_tab3_latent_drift import latent_drift_simulation

from helpers.helpers_tab3_trajectory import (pad_truncate, pad_truncate_2d, center_vectors, normalize_rows, prepare_trajectory )
from helpers.helpers_tab3_trajectory import (prepare_user_trajectory, plot_top10_markers, plot_category_scatter)

from helpers.helpers_tab3_knn import plot_category_biased_knn_graph, plot_user_political_extremeness_tab3


# Optional: ALS
try:
    from implicit.als import AlternatingLeastSquares
    import scipy.sparse as sp
    ALS_AVAILABLE = True
except:
    ALS_AVAILABLE = False

# ------------------------
# Load Data (once globally)
# ------------------------
original_df = pd.read_csv("hands-on-activity/streamlit_code/merged_movielens.csv")

merged_df = pd.read_csv("hands-on-activity/streamlit_code/movielens_100k_categories.csv")
merged_df.columns = merged_df.columns.str.strip()

user_cols = ["userId", "age", "gender", "occupation", "zip_code"]
users = pd.read_csv("hands-on-activity/data/ml-100k/u.user", sep="|", names=user_cols, encoding="latin-1")
users = users.drop(columns=["zip_code"])
users.columns = users.columns.str.strip()

movie_info_cols = ["movieId", "title", "category"]
movie_info = merged_df[movie_info_cols].drop_duplicates()
movie_info_reset = movie_info.reset_index(drop=True)

st.set_page_config(page_title="SVD Recommendation System", layout="wide")

# ------------------------
# Sidebar (global)
# ------------------------
st.sidebar.header("Global Parameters:")

# Select subset size
n_users = st.sidebar.slider(
    "Number of users to include",
    5, len(merged_df['userId'].unique()), 100
)
n_movies = st.sidebar.slider(
    "Number of movies to include",
    10, len(merged_df['movieId'].unique()), 300
)

## Select user and method
user_options = sorted(merged_df['userId'].unique()[:n_users])

selected_user = st.sidebar.selectbox(
    "Select user for recommendations / simulation",
    user_options
)


method = st.sidebar.selectbox(
    "Recommendation Method / Latent Model",
    ["SVD", "ALS", "PMF"]
)

# ------------------------------------------------------------
# Dynamically determine SAFE maximum k based on method + data
# ------------------------------------------------------------
hard_limit = min(n_users, n_movies)

if method == "ALS":
    practical_limit = max(3, min(hard_limit, hard_limit // 2))
elif method == "PMF":
    practical_limit = max(3, min(hard_limit, 50))
else:
    practical_limit = hard_limit

max_k = practical_limit

latent_factors = st.sidebar.slider(
    "Latent factors (k)",
    min_value=3,
    max_value=max_k,
    value=min(10, max_k),
    step=1,
    help=f"Max k = {max_k} (model: {method}, users={n_users}, movies={n_movies})"
)

# Simulation controls
learning_rate = st.sidebar.slider("Simulation learning rate", 0.05, 0.5, 0.3)

# ------------------------
# Subset Data
# ------------------------
user_ids = merged_df['userId'].unique()[:n_users]
movie_ids = merged_df['movieId'].unique()[:n_movies]

merged_df_sub = merged_df[
    merged_df['userId'].isin(user_ids) &
    merged_df['movieId'].isin(movie_ids)
]

df_display = merged_df_sub.merge(users, on="userId", how="left")

# Computational & UI matrices (kept separate)
R_full = df_display.pivot(index='userId', columns='movieId',
                          values='rating').fillna(0)
R_display = df_display.pivot(index='userId', columns='movieId',
                             values='rating')

als_model = None
U_pmf = None
V_pmf = None

# ------------------------
# Tabs
# ------------------------
tab1, tab2, tab3 = st.tabs(["Data", "Recommendations", "Simulation"])

# ------------------------
# Tab 1: Data Display
# ------------------------
with tab1:
    st.header("Movielens Subset with User and Movie Info")

    col1, col2  = st.columns(2)

    with col1: 

        st.subheader("Original Dataset ")
        st.markdown(f"**Table size:** {original_df.shape[0]} rows × {original_df.shape[1]} columns")
        st.dataframe(original_df)

    with col2:
        st.subheader("Category Mapping:")
        st.markdown("19 Movie Genres mapped to 3 Political Categories")

        mapping_df = get_category_mapping_df()
        st.dataframe(mapping_df)


    # Count selected users and movies
    n_users_selected = len(user_ids)
    n_movies_selected = len(movie_ids)

    st.subheader(f"Ratings Table — {n_users_selected} users × {n_movies_selected} movies")

    cols_to_show = ["userId", "age", "gender", "occupation",
                    "movieId", "title", "category", "rating"]
    existing_cols = [c for c in cols_to_show if c in df_display.columns]

    df_to_display = df_display[existing_cols]

    col1, col2  = st.columns(2)

    with col1: 
        st.markdown(f"**Table size:** {df_to_display.shape[0]} rows × {df_to_display.shape[1]} columns")
        st.dataframe(df_to_display.style.format(na_rep=""))

    with col2:
        st.subheader("Category Distribution")

        color_map = {
            "neutral": "green",
            "mildly_political": "orange",
            "extreme": "red"
        }

        # compute counts
        cat_counts_full = merged_df["category"].value_counts().reset_index()
        cat_counts_full.columns = ["category", "count"]

        cat_counts_filtered = df_to_display["category"].value_counts().reset_index()
        cat_counts_filtered.columns = ["category", "count"]

        # Use helper to draw the two charts
        plot_category_distribution(cat_counts_full, cat_counts_filtered, color_map)


    # User–Item Matrix
    st.subheader("User–Item Matrix")
    st.markdown(f"**Matrix shape:** {R_full.shape[0]} users × {R_full.shape[1]} movies")

    R_display_clean = R_full.replace(0, np.nan)
    st.dataframe(R_display_clean)

    # Tab1: User Political Extremeness from actual ratings
    fig_tab1 = plot_user_political_extremeness(
        df_display=df_display,           # actual ratings
        selected_user=selected_user,
        title="User Political Extremeness (Actual Ratings)"
    )
    st.plotly_chart(fig_tab1, use_container_width=True, key=f"user_extremeness_tab1_{selected_user}")




# -------------------------------------------------------
# Tab 2: Matrix Factorization and Latent Space Exploration
# --------------------------------------------------------
with tab2:

    st.header("Matrix Factorization and Latent Space Exploration")

    # ------------------------
    # Prepare rating matrix for computation
    # ------------------------
    R_matrix, R_actual, users_idx, movies_idx = build_rating_matrices(df_display)


    # ------------------------
    # Factorization
    # ------------------------
    if method == "SVD":
        user_latent, movie_latent, pred_ratings, s = run_svd(R_matrix, latent_factors)

    elif method == "ALS":
        user_latent, movie_latent, pred_ratings = run_als(
        R_matrix, 
        k=latent_factors, 
        reg=0.5, 
        iterations=50,
        clip_ratings=(1,5)
    )
        s = None

    elif method == "PMF":
        user_latent, movie_latent, pred_ratings = run_pmf(R_matrix, latent_factors)
        s = None

    else:
        st.error("Unknown method.")
        st.stop()

    # ------------------------
    # Latent Matrices Display
    # ------------------------
   
    if method == "SVD":
        show_svd_block(user_latent, movie_latent, s, latent_factors, users_idx, movies_idx)
    elif method == "ALS":
        show_als_block(user_latent, movie_latent, users_idx, movies_idx)
    elif method == "PMF":
        show_pmf_block(user_latent, movie_latent, users_idx, movies_idx)

   
    st.markdown("### Reconstructed (predicted) ratings")
        
    # Convert pred_ratings to a labeled DataFrame with user_id and movie_id
    pred_df = pd.DataFrame(pred_ratings, index=users_idx, columns=movies_idx)
    pred_df.index.name = "user_id"
    pred_df.columns.name = "movie_id"

    st.dataframe(pred_df)

    st.subheader("Prediction Error Metrics")
    # --- Compute RMSE and MSE for SVD reconstruction ---

    # Create a mask of known ratings (non-NaN entries)
    valid_mask = ~np.isnan(R_actual)

    # Compute squared errors only on observed ratings
    squared_errors = (R_matrix[valid_mask] - pred_ratings[valid_mask]) ** 2

    # Mean Squared Error
    mse = np.mean(squared_errors)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Display with helper
    show_error_metrics(mse, rmse)


    st.subheader(f"Predicted Ratings for Unrated Movies - User {selected_user}")

    col1, col2 = st.columns(2)
    with col1: 
        # ------------------------
        # Predicted ratings for selected user (only unrated movies)
        # ------------------------
        
        user_idx = users_idx.index(selected_user)
        pred_user_ratings = pred_ratings[user_idx, :]

        # Build DataFrame for all movies
        user_pred_df = pd.DataFrame({
            "movieId": movies_idx,
            "predicted_rating": pred_user_ratings
        }).merge(
            movie_info_reset[["movieId", "title", "category"]],
            on="movieId",
            how="left"
        )

        # Get actual ratings (keep NaN for unrated)
        actual_ratings = R_actual.loc[selected_user, movies_idx]
        user_pred_df["previous_rating"] = actual_ratings.values

        # Keep only unrated movies
        user_pred_df = user_pred_df[user_pred_df["previous_rating"].isna()]

        # Sort by predicted rating descending
        user_pred_sorted = user_pred_df.sort_values("predicted_rating", ascending=False)
        st.dataframe(user_pred_sorted.head(10))

    with col2:
        st.subheader("Category Distribution (Top 10 Recommendations)")

        # Color coding
        color_map = {
            "neutral": "green",
            "mildly_political": "orange",
            "extreme": "red"
        }

        # Take the top 10 recommended movies
        top10 = user_pred_sorted.head(10)

        # Count categories in top 10
        cat_counts_top10 = (
            top10["category"]
            .value_counts()
            .reset_index()
        )
        cat_counts_top10.columns = ["category", "count"]

        plot_top10_category_bar(cat_counts_top10, selected_user)


    # ------------------------
    # 3D Latent Space (User-Centered & Political Category Clusters)
    # ------------------------
    st.subheader("3D Latent Space – User-Centered ")
    plot_latent_3d(
        user_latent, movie_latent, user_idx,
        movies_idx, movie_info_reset,
        user_pred_sorted, method, latent_factors, selected_user
    )

   # Convert predicted ratings to long-form DataFrame
    pred_df_full = pd.DataFrame(pred_ratings, index=users_idx, columns=movies_idx)
    pred_df_full.index.name = "userId"
    pred_df_full = pred_df_full.reset_index().melt(id_vars="userId", var_name="movieId", value_name="rating")

    # Merge category info
    pred_df_full = pred_df_full.merge(
        movie_info_reset[["movieId", "category"]],
        on="movieId",
        how="left"
    )

    # Plot predicted extremeness
    fig_tab2 = plot_user_political_extremeness(
        df_display=pred_df_full,        # predicted ratings
        selected_user=selected_user,
        title=f"Predicted Ratings: User Political Extremeness ({method})"
    )

    st.plotly_chart(fig_tab2, use_container_width=True, key=f"user_extremeness_tab2_{method}_{selected_user}")






# ------------------------
# Tab 3: Latent Drift Simulation
# ------------------------
with tab3:
    st.header(f"Latent Drift Simulation for User {selected_user} ({method})")

    # ------------------------
    # 0. Compute latent drift (returns U_scaled, V_scaled, V_mean, movie_info_safe, pred_df, top10, k)
    # ------------------------
    drift_data = latent_drift_simulation(
        R_full, R_display, movie_info_reset,
        selected_user, method, latent_factors, learning_rate, ALS_AVAILABLE
    )

    # Unpack
    U_scaled = drift_data["U_scaled"]
    V_scaled = drift_data["V_scaled"]
    V_mean = drift_data["V_mean"]
    movie_info_safe = drift_data["movie_info_safe"]
    pred_df = drift_data["pred_df"]
    top10 = drift_data["top10"]
    k = drift_data["k"]

    # ------------------------
    # 1. Safe movie info defaults
    # ------------------------
    movie_info_safe = movie_info_reset.copy()
    if "movieId" not in movie_info_safe.columns:
        movie_info_safe["movieId"] = list(R_full.columns)
    if "title" not in movie_info_safe.columns:
        movie_info_safe["title"] = movie_info_safe["movieId"].astype(str)
    if "category" not in movie_info_safe.columns:
        movie_info_safe["category"] = "neutral"

    # ------------------------
    # 2. Initialize user vector & trajectory
    # ------------------------
    if "user_latent_current" not in st.session_state or st.session_state.user_latent_current is None:
        # safe fallback: if U_scaled smaller than expected, pad
        st.session_state.user_latent_current = pad_truncate(U_scaled[user_idx, :].copy(), k)

    st.session_state.user_latent_current = pad_truncate(st.session_state.user_latent_current, k)

    if "trajectory" not in st.session_state or not st.session_state.trajectory:
        st.session_state.trajectory = [st.session_state.user_latent_current.copy()]

    current_user_vec = st.session_state.user_latent_current.copy()

    # ------------------------
    # 3. Safe predictions (length aligned)
    # ------------------------
    pred_user_ratings = np.ravel(current_user_vec @ V_scaled.T)
    all_movie_ids = list(R_full.columns)
    if len(pred_user_ratings) != len(all_movie_ids):
        pad_val = np.nanmean(pred_user_ratings) if pred_user_ratings.size > 0 else 0.0
        pred_user_ratings = pad_truncate(pred_user_ratings, len(all_movie_ids), fill_val=pad_val)

    pred_df = pd.DataFrame({"movieId": all_movie_ids, "pred_rating": pred_user_ratings})
    rated_movies = R_display.loc[selected_user].dropna().index.tolist() if selected_user in R_display.index else []
    pred_df = pred_df[~pred_df["movieId"].isin(rated_movies)].reset_index(drop=True)
    top10 = pred_df.sort_values("pred_rating", ascending=False).head(10)
    top10 = top10.merge(movie_info_safe[["movieId", "title", "category"]], on="movieId", how="left")
    # ensure category exists
    if "category" not in top10.columns:
        top10["category"] = "neutral"

    # ------------------------
    # callback functions used by buttons (ALS-safe on_click)
    # ------------------------
    def _simulate_interaction_callback(movie_id: int, title: str, lr: float, k_local: int):
        """
        Update user latent current and trajectory in session state.
        Must NOT call st.write or other Streamlit display functions here.
        """
        # guard: ensure user vector exists
        if "user_latent_current" not in st.session_state or st.session_state.user_latent_current is None:
            return

        # find movie index in stable all_movie_ids
        try:
            movie_idx_local = all_movie_ids.index(movie_id)
        except ValueError:
            st.session_state["_last_sim_failure"] = f"movie_id {movie_id} not found"
            return

        # update
        before_vec = st.session_state.user_latent_current.copy()
        st.session_state.user_latent_current += lr * (V_scaled[movie_idx_local, :] - st.session_state.user_latent_current)
        st.session_state.user_latent_current = pad_truncate(st.session_state.user_latent_current, k_local)

        # append trajectory
        if "trajectory" not in st.session_state or not st.session_state.trajectory:
            st.session_state.trajectory = []
        st.session_state.trajectory.append(st.session_state.user_latent_current.copy())

        # store a short payload for UI to display after rerun
        st.session_state["_last_sim_clicked"] = {
            "movieId": movie_id,
            "title": title,
            "prev_vector": before_vec
        }

    def _recovery_callback(lr: float, k_local: int):
        if "user_latent_current" not in st.session_state or st.session_state.user_latent_current is None:
            return
        neutral_mask = movie_info_safe["category"] == "neutral"
        neutral_vec = V_scaled[neutral_mask, :].mean(axis=0) if neutral_mask.any() else V_mean
        before_vec = st.session_state.user_latent_current.copy()
        st.session_state.user_latent_current += lr * (neutral_vec - st.session_state.user_latent_current)
        st.session_state.user_latent_current = pad_truncate(st.session_state.user_latent_current, k_local)
        if "trajectory" not in st.session_state or not st.session_state.trajectory:
            st.session_state.trajectory = []
        st.session_state.trajectory.append(st.session_state.user_latent_current.copy())
        st.session_state["_last_recovery"] = {"prev_vector": before_vec, "title": "Neutral movie"}

    # ------------------------
    # 4. Clickable top-10 buttons (ALS-safe with on_click)
    # ------------------------
    st.subheader("Click one of the top-10 recommended movies to simulate interaction")

    emoji_map = {"neutral": "🟢", "mildly_political": "🟠", "extreme": "🔴"}
    category_colors = {"neutral": "green", "mildly_political": "orange", "extreme": "red"}

    # produce a stable top10 ordering (by pred_rating then movieId)
    top10_sorted = top10.sort_values(["pred_rating", "movieId"], ascending=[False, True]).head(10).reset_index(drop=True)

    # ensure trajectory exists
    if "trajectory" not in st.session_state or not st.session_state.trajectory:
        st.session_state.trajectory = [st.session_state.user_latent_current.copy()]

    # layout columns
    cols_row1 = st.columns(5)
    cols_row2 = st.columns(5)

    for idx, row in top10_sorted.iterrows():
        emo = emoji_map.get(row.get("category", "neutral"), "⚪")
        title = row.get("title", str(row["movieId"]))
        short_title = title if len(title) <= 35 else title[:32] + "..."
        label = f"{emo} {short_title}"

        # stable key by movieId
        key = f"sim_btn_movie_{int(row['movieId'])}"
        container = cols_row1[idx] if idx < 5 else cols_row2[idx - 5]

        # attach on_click callback with stable args
        container.button(
            label,
            key=key,
            help=f"Category: {row.get('category','neutral')}",
            on_click=_simulate_interaction_callback,
            args=(int(row["movieId"]), title, float(learning_rate), int(k)),
        )

    # show confirmation if callback fired
    if "_last_sim_clicked" in st.session_state:
        info = st.session_state["_last_sim_clicked"]
        st.success(f"Simulated interaction: {info['title']}")
        # remove so it doesn't repeat across reruns
        del st.session_state["_last_sim_clicked"]

    if "_last_sim_failure" in st.session_state:
        st.warning(st.session_state["_last_sim_failure"])
        del st.session_state["_last_sim_failure"]

    # ------------------------
    # 5. Recovery Action (use on_click as well)
    # ------------------------
    st.button("🎬 Watch a Neutral Movie (Recovery Action)", on_click=_recovery_callback, args=(float(learning_rate), int(k)))
    if "_last_recovery" in st.session_state:
        st.warning("You watched a neutral movie — your preferences shift back toward balance.")
        del st.session_state["_last_recovery"]

    # ------------------------
    # 6. 3D Latent Space Plot (User + Trajectory)
    # ------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("3D Latent Space – User-Centered")
        if k < 3:
            st.warning("Need at least 3 latent factors for 3D visualization.")
        else:
            # Normalize scales to make U and V comparable across methods
            scale_factor = max(np.std(U_scaled), np.std(V_scaled), 1e-8)
            V_norm = V_scaled / scale_factor

            # prepare movie coordinates truncated to 3 dims
            movie_coords = pad_truncate_2d(V_norm, 3)
            movie_df_plot = pd.DataFrame(movie_coords, columns=["dim1", "dim2", "dim3"])
            movie_df_plot["movieId"] = all_movie_ids
            movie_df_plot = movie_df_plot.merge(movie_info_safe[["movieId", "title", "category"]], on="movieId", how="left")

            # top-10 overlay (preserve order)
            top10_plot = movie_df_plot[movie_df_plot["movieId"].isin(top10_sorted["movieId"])]
            top10_plot["order"] = top10_plot["movieId"].map({m: i for i, m in enumerate(top10_sorted["movieId"].tolist())})
            top10_plot = top10_plot.sort_values("order")

            # trajectory (normalized to same scale)
            traj_norm = np.array([pad_truncate(u, k) for u in st.session_state.trajectory]) / scale_factor
            traj3 = pad_truncate_2d(traj_norm, 3)

            # center movies so current user is at origin (use current user vector normalized & truncated)
            user_vec_norm = pad_truncate_2d(st.session_state.user_latent_current.reshape(1, -1), 3)[0] / scale_factor
            movie_df_plot[["dim1", "dim2", "dim3"]] = center_vectors(movie_df_plot[["dim1", "dim2", "dim3"]].values, user_vec_norm)
            top10_plot[["dim1", "dim2", "dim3"]] = center_vectors(top10_plot[["dim1", "dim2", "dim3"]].values, user_vec_norm)
            traj_centered = traj3 - user_vec_norm  # previous positions relative to current user

            # Plot
            fig = go.Figure()
            plot_category_scatter(fig, movie_df_plot, category_colors)
            plot_top10_markers(fig, top10_plot, category_colors, size=9)

            # previous user positions (small markers) + line
            if traj_centered.shape[0] > 1:
                fig.add_trace(go.Scatter3d(
                    x=traj_centered[:, 0],
                    y=traj_centered[:, 1],
                    z=traj_centered[:, 2],
                    mode="lines+markers+text",
                    name="User Trajectory",
                    line=dict(color="blue", width=4),
                    marker=dict(size=6, color="blue"),
                    text=[f"Step {i}" for i in range(traj_centered.shape[0])],
                    textposition="bottom center"
                ))

            # previous small marker (if there is at least one previous step)
            if traj_centered.shape[0] > 1:
                prev = traj_centered[-2]  # penultimate = previous
                fig.add_trace(go.Scatter3d(
                    x=[prev[0]], y=[prev[1]], z=[prev[2]],
                    mode="markers",
                    name="Previous User",
                    marker=dict(size=6, color="lightblue", symbol="circle"),
                    showlegend=True
                ))

            # big marker for current user at origin
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers+text",
                name=f"User {selected_user}",
                marker=dict(size=10, color="blue", symbol="circle"),
                text=[f"User {selected_user}"],
                textposition="top center"
            ))

            fig.update_layout(
                title=f"3D Latent Space – {method} (k={k}) [User-Centered]",
                scene=dict(xaxis_title="Latent Dim 1", yaxis_title="Latent Dim 2", zaxis_title="Latent Dim 3"),
                height=750, legend=dict(x=0.02, y=0.98)
            )

            st.plotly_chart(fig, use_container_width=True)

    # ------------------------
    # 7. Enhanced UMAP Projection
    # ------------------------
    with col2:
        try:
            umap_key = f"umap_{method}_{n_movies}_{k}"
            if umap_key not in st.session_state:
                V_center = V_scaled - V_mean
                V_umap_input = normalize_rows(V_center)
                umap_model = UMAP(n_components=3, random_state=42, metric="cosine")
                movie_coords_umap = umap_model.fit_transform(V_umap_input)
                st.session_state[umap_key] = {"model": umap_model, "V_mean": V_mean}
            else:
                umap_model = st.session_state[umap_key]["model"]
                V_mean_stored = st.session_state[umap_key]["V_mean"]
                V_umap_input = normalize_rows(V_scaled - V_mean_stored)
                movie_coords_umap = umap_model.transform(V_umap_input)

            movie_umap_df = pd.DataFrame(movie_coords_umap, columns=["dim1", "dim2", "dim3"])
            movie_umap_df["movieId"] = all_movie_ids
            movie_umap_df = movie_umap_df.merge(movie_info_safe[["movieId", "title", "category"]], on="movieId", how="left")

            traj_umap = normalize_rows(prepare_trajectory(st.session_state.trajectory, k, V_mean) - st.session_state[umap_key]["V_mean"])
            user_umap_df = pd.DataFrame(umap_model.transform(traj_umap), columns=["dim1", "dim2", "dim3"])
            user_umap_df["step"] = np.arange(len(user_umap_df))

            fig_umap = go.Figure()
            plot_category_scatter(fig_umap, movie_umap_df, category_colors)
            plot_top10_markers(fig_umap, movie_umap_df[movie_umap_df["movieId"].isin(top10_sorted["movieId"])], category_colors, size=9)

            if len(user_umap_df) > 1:
                fig_umap.add_trace(go.Scatter3d(
                    x=user_umap_df["dim1"], y=user_umap_df["dim2"], z=user_umap_df["dim3"],
                    mode="lines+markers+text",
                    name="User Trajectory (UMAP)",
                    line=dict(color="blue", width=4),
                    marker=dict(size=6, color="blue"),
                    text=[f"Step {i}" for i in user_umap_df["step"]],
                    textposition="bottom center"
                ))

            # current user (UMAP)
            cur = user_umap_df.iloc[-1]
            fig_umap.add_trace(go.Scatter3d(
                x=[cur["dim1"]], y=[cur["dim2"]], z=[cur["dim3"]],
                mode="markers+text",
                name="Current User Position",
                marker=dict(size=10, color="blue", symbol="circle"),
                text=[f"User {selected_user}"], textposition="top center"
            ))

            fig_umap.update_layout(
                title="Enhanced UMAP Projection (Cosine distance): Content Clusters + User Drift",
                scene=dict(xaxis_title="UMAP Dim 1", yaxis_title="UMAP Dim 2", zaxis_title="UMAP Dim 3"),
                height=750, legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_umap, use_container_width=True)
        except Exception as e:
            st.warning(f"UMAP projection skipped: {e}")

    col1, col2 = st.columns(2)

    with col1:
        # ------------------------
        # 8. Radicalization Index and Category Distribution
        # ------------------------
        if "category" not in top10.columns:
            top10["category"] = "neutral"
        rad_idx = float((top10["category"] == "extreme").mean())
        delta_val = rad_idx - st.session_state.get("prev_rad_idx", 0.0)
        st.metric("Radicalization Index", f"{rad_idx*100:.1f}%", delta=f"{delta_val*100:+.1f}%")
        st.session_state.prev_rad_idx = rad_idx

        cat_counts = top10["category"].value_counts().reindex(["neutral", "mildly_political", "extreme"], fill_value=0)
        cat_chart = px.bar(
            x=cat_counts.index, y=cat_counts.values,
            labels={"x": "Category", "y": "Count"},
            title="Distribution of Top-10 Recommendations",
            color=cat_counts.index,
            color_discrete_map=category_colors
        )
        st.plotly_chart(cat_chart, use_container_width=True)

    with col2: 

        # ------------------------
        # 9. Category-Biased KNN Graph (Cluster Representation)
        # ------------------------
        st.subheader("Category-Biased KNN Graph (Top-10 Highlight + User Drift)")

        try:
            # --------------------------------------------------
            # IMPORTANT: KNN GRAPH USES A CLUSTER-FIRST FRAME
            # - NO user-centering of movies
            # - NO global variance normalization
            # - Spring layout defines geometry
            # --------------------------------------------------

            # --- Movie coordinates (cluster-first, NOT user-centered) ---
            movie_coords_knn = pad_truncate_2d(V_scaled, 3)

            movie_umap_df_knn = pd.DataFrame(
                movie_coords_knn,
                columns=["dim1", "dim2", "dim3"]
            )
            movie_umap_df_knn["movieId"] = all_movie_ids
            movie_umap_df_knn = movie_umap_df_knn.merge(
                movie_info_safe[["movieId", "title", "category"]],
                on="movieId",
                how="left"
            )

            # --- Top-10 subset (stable order preserved) ---
            top10_knn = movie_umap_df_knn[
                movie_umap_df_knn["movieId"].isin(top10_sorted["movieId"])
            ].copy()

            top10_knn["order"] = top10_knn["movieId"].map(
                {m: i for i, m in enumerate(top10_sorted["movieId"].tolist())}
            )
            top10_knn = top10_knn.sort_values("order")

            # --------------------------------------------------
            # User trajectory (visual exaggeration only)
            # --------------------------------------------------
            traj_knn = np.array([pad_truncate(u, k) for u in st.session_state.trajectory])
            traj_knn_3d = pad_truncate_2d(traj_knn, 3)

            # Exaggerate trajectory so drift is visible
            TRAJECTORY_VISUAL_SCALE = 8.0  # safe range: 5–12
            # express trajectory as offsets from starting point
            traj_offsets = traj_knn_3d - traj_knn_3d[0]

            # exaggerate offsets only
            traj_knn_3d = traj_offsets * TRAJECTORY_VISUAL_SCALE

            user_umap_df_knn = pd.DataFrame(
                traj_knn_3d,
                columns=["dim1", "dim2", "dim3"]
            )
            user_umap_df_knn["step"] = np.arange(len(user_umap_df_knn))

            # --------------------------------------------------
            # Render KNN graph (spring layout defines clusters)
            # --------------------------------------------------
            fig_knn = plot_category_biased_knn_graph(
                movie_umap_df=movie_umap_df_knn,
                user_umap_df=user_umap_df_knn,
                top10=top10_knn,
                category_colors=category_colors,

                # --- Stronger defaults for demonstration ---
                K=6,
                bias_strength=1.3,
                exaggeration=1.15,
                drift_strength=0.4,
                spring_scale=6.0,
                spring_k=0.25
            )

            st.plotly_chart(fig_knn, use_container_width=True)

        except Exception as e:
            st.warning(f"Category-biased graph skipped: {e}")

    # ------------------------
    # Political Extremeness Plot (Predicted / Current User, Top-10 Only)
    # ------------------------

    # 1. Prepare top-10 IDs for the current user
    top10_sorted = top10.sort_values(["pred_rating", "movieId"], ascending=[False, True]).head(10).reset_index(drop=True)
    top10_ids = top10_sorted["movieId"].tolist()

    # 2. Prepare static data for other users (compute once)
    if "all_users_static_top10" not in st.session_state:
        # Filter original ratings to top-10 movies only
        df_static_top10 = df_display[df_display["movieId"].isin(top10_ids)].copy()
        
        # Ensure all categories exist
        if "category" not in df_static_top10.columns:
            df_static_top10["category"] = "neutral"
        
        st.session_state.all_users_static_top10 = df_static_top10
    else:
        df_static_top10 = st.session_state.all_users_static_top10

    # 3. Prepare predicted ratings for selected user (top-10 only)
    pred_user_top10 = pred_user_ratings[[all_movie_ids.index(m) for m in top10_ids]]
    df_selected_user_top10 = pd.DataFrame({
        "userId": selected_user,
        "movieId": top10_ids,
        "rating": pred_user_top10
    }).merge(movie_info_safe[["movieId", "category"]], on="movieId", how="left")

    # 4. Merge static users + selected user
    df_all_pred_top10 = pd.concat(
        [df_static_top10, df_selected_user_top10],
        ignore_index=True
    )

    # 5. Plot extremeness (2D scatter)
    fig_extremeness = plot_user_political_extremeness_tab3(
        df_display=df_all_pred_top10,
        selected_user=selected_user,
        title=f"User Political Extremeness (Top-10) – Step {len(st.session_state.trajectory)}"
    )

    st.plotly_chart(
        fig_extremeness,
        use_container_width=True,
        key=f"user_extremeness_top10_tab3_{method}_{selected_user}_{len(st.session_state.trajectory)}"
    )
