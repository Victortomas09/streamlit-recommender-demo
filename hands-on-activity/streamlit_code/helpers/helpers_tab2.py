# helpers_tab2.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp


# --------------------------------------------------------------------
# 1. Matrix creation helpers
# --------------------------------------------------------------------
def build_rating_matrices(df_display):
    """Return (R_matrix_zero_filled, R_actual_nan_preserved, users_idx, movies_idx)."""
    R = df_display.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    R_matrix = R.values
    users_idx = list(R.index)
    movies_idx = list(R.columns)
    R_actual = df_display.pivot(index='userId', columns='movieId', values='rating')
    return R_matrix, R_actual, users_idx, movies_idx


# --------------------------------------------------------------------
# 2. Factorization helpers
# --------------------------------------------------------------------
def run_svd(R_matrix, k):
    U, s, Vt = np.linalg.svd(R_matrix, full_matrices=False)
    k_use = min(k, U.shape[1])
    U_k = U[:, :k_use]
    S_k = np.diag(s[:k_use])
    Vt_k = Vt[:k_use, :]
    sqrt_S = np.diag(np.sqrt(s[:k_use]))
    user_latent = U_k @ sqrt_S
    item_latent = Vt_k.T @ sqrt_S
    pred = user_latent @ item_latent.T
    return user_latent, item_latent, pred, s


def run_als(R_matrix, k=10, reg=0.5, iterations=50, clip_ratings=(1,5)):
    """
    Explicit ALS for explicit ratings (robust for small, sparse matrices).

    Args:
        R_matrix: np.array, shape (n_users, n_items), missing ratings = 0
        k: int, number of latent factors
        reg: float, regularization parameter
        iterations: int, number of ALS iterations
        clip_ratings: tuple (min_rating, max_rating), to clip predictions

    Returns:
        U: user latent factors (n_users x k)
        V: item latent factors (n_items x k)
        pred: predicted rating matrix (n_users x n_items)
    """

    R = R_matrix.copy().astype(float)
    n_users, n_items = R.shape

    # Mask of observed entries
    mask = (R > 0).astype(float)

    # Initialize latent factors
    U = np.random.normal(scale=0.1, size=(n_users, k))
    V = np.random.normal(scale=0.1, size=(n_items, k))

    # Precompute regularization matrix
    regI = reg * np.eye(k)

    for _ in range(iterations):

        # Update user factors U
        for u in range(n_users):
            idx = mask[u] > 0
            if not np.any(idx):
                continue
            V_u = V[idx]
            R_u = R[u, idx]
            A = V_u.T @ V_u + regI
            b = V_u.T @ R_u
            U[u] = np.linalg.solve(A, b)

        # Update item factors V
        for i in range(n_items):
            idx = mask[:, i] > 0
            if not np.any(idx):
                continue
            U_i = U[idx]
            R_i = R[idx, i]
            A = U_i.T @ U_i + regI
            b = U_i.T @ R_i
            V[i] = np.linalg.solve(A, b)

    # Predicted ratings
    pred = U @ V.T

    # Clip predictions to rating range
    pred = np.clip(pred, clip_ratings[0], clip_ratings[1])

    return U, V, pred



def run_pmf(R_matrix, k, lr=0.01, reg=0.1, epochs=50):
    R = R_matrix.astype(float)
    R[R == 0] = np.nan
    n_users, n_items = R.shape

    U = np.random.normal(scale=0.1, size=(n_users, k))
    V = np.random.normal(scale=0.1, size=(n_items, k))

    obs = np.argwhere(~np.isnan(R))

    for _ in range(epochs):
        np.random.shuffle(obs)
        for i, j in obs:
            r = R[i, j]
            ui, vj = U[i].copy(), V[j].copy()
            err = r - ui @ vj
            U[i] += lr * (err * vj - reg * ui)
            V[j] += lr * (err * ui - reg * vj)

    pred = U @ V.T
    return U, V, pred


# --------------------------------------------------------------------
# 3. Plot helpers
# --------------------------------------------------------------------
def plot_top10_category_bar(cat_counts_top10, selected_user):
    color_map = {"neutral": "green", "mildly_political": "orange", "extreme": "red"}

    fig = px.bar(
        cat_counts_top10,
        x="category",
        y="count",
        title=f"User {selected_user} Top 10 Recommendations",
        color="category",
        color_discrete_map=color_map,
        height=300,
        width=400,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_latent_3d(user_latent, movie_latent, user_idx, movies_idx, movie_info_reset,
                   user_pred_sorted, method, k, selected_user):
    if user_latent.shape[1] < 3:
        st.warning("Need at least 3 latent factors for 3D.")
        return

    user_latent_norm = user_latent / np.linalg.norm(user_latent, axis=1, keepdims=True)
    movie_latent_norm = movie_latent / np.linalg.norm(movie_latent, axis=1, keepdims=True)

    user_vec = user_latent_norm[user_idx, :3]
    movie_centered = movie_latent_norm[:, :3] - user_vec
    user_center = np.zeros(3)

    movie_df = pd.DataFrame(movie_centered, columns=["dim1","dim2","dim3"])
    movie_df["movieId"] = movies_idx
    movie_df = movie_df.merge(movie_info_reset, on="movieId", how="left")

    top10_ids = user_pred_sorted.head(10).movieId.tolist()
    top10_subset = movie_df[movie_df["movieId"].isin(top10_ids)]

    colors = {"neutral": "green", "mildly_political": "orange", "extreme": "red"}
    fig = go.Figure()

    for cat, color in colors.items():
        subset = movie_df[movie_df["category"]==cat]
        fig.add_trace(go.Scatter3d(
            x=subset["dim1"], y=subset["dim2"], z=subset["dim3"],
            mode='markers',
            name=f"{cat}",
            marker=dict(size=6, color=color, opacity=0.7),
            text=subset["title"],
        ))

    fig.add_trace(go.Scatter3d(
        x=top10_subset["dim1"], y=top10_subset["dim2"], z=top10_subset["dim3"],
        mode='markers+text',
        name="Top-10",
        marker=dict(
            size=10,
            color=[colors.get(c, "gray") for c in top10_subset["category"]],
            symbol="diamond",
            opacity=1,
        ),
        text=top10_subset["title"],
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        name=f"User {selected_user}",
        marker=dict(size=10, color="blue"),
        text=[f"User {selected_user}"],
    ))

    fig.update_layout(
        title=f"3D Latent Space – {method} (k={k})",
        scene=dict(xaxis_title='Dim1', yaxis_title='Dim2', zaxis_title='Dim3'),
        height=750,
    )

    st.plotly_chart(fig, use_container_width=True)
