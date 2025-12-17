import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from scipy.sparse import spmatrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.decomposition import NMF
from scipy.linalg import svd

def latent_drift_simulation(
    R_full, R_display, movie_info_reset, selected_user,
    method, latent_factors, learning_rate, ALS_AVAILABLE=True
):
    """
    Run the latent drift simulation with SVD, PMF/NMF, or explicit ALS.
    Returns latent matrices, predicted ratings, top-10 recommendations, and movie info.
    """
    import numpy as np
    from sklearn.decomposition import NMF
    from numpy.linalg import svd

    n_users, n_movies = R_full.shape
    k = latent_factors
    user_idx = list(R_full.index).index(selected_user)

    # ------------------------
    # Session state reset
    # ------------------------
    reset_needed = (
        st.session_state.get("prev_latent_factors") != k
        or st.session_state.get("prev_method") != method
        or st.session_state.get("prev_user") != selected_user
        or st.session_state.get("prev_lr") != learning_rate
        or st.session_state.get("prev_n_movies") != n_movies
    )
    if reset_needed:
        st.session_state.user_latent_current = None
        st.session_state.clicked_movie = None
        st.session_state.prev_rad_idx = 0.0
        st.session_state.trajectory = []
        st.session_state.prev_latent_factors = k
        st.session_state.prev_method = method
        st.session_state.prev_user = selected_user
        st.session_state.prev_lr = learning_rate
        st.session_state.prev_n_movies = n_movies
    if "prev_rad_idx" not in st.session_state:
        st.session_state.prev_rad_idx = 0.0

    # ------------------------
    # Compute latent factors
    # ------------------------
    if method == "SVD":
        U_sim, s_sim, Vt_sim = svd(R_full.values, full_matrices=False)
        k_sim = min(k, U_sim.shape[1])
        U_sim = U_sim[:, :k_sim]
        V_sim = Vt_sim[:k_sim, :].T

    elif method == "PMF":
        nmf_model = NMF(n_components=k, init="random", random_state=42, max_iter=200)
        U_sim = nmf_model.fit_transform(R_full.values)
        V_sim = nmf_model.components_.T

    elif method == "ALS":
        # Use your new explicit ALS
        from helpers.helpers_tab2 import run_als  # adjust import path as needed
        U_sim, V_sim, pred_matrix = run_als(
            R_full.values, k=k, reg=0.5, iterations=50, clip_ratings=(1,5)
        )
    else:
        # fallback: zeros
        U_sim = np.zeros((n_users, k))
        V_sim = np.zeros((n_movies, k))

    # ------------------------
    # Scale latent space
    # ------------------------
    # Use both user + movie latent factors to determine scale
    all_latents = np.vstack([U_sim, V_sim])
    scale_factor = np.std(all_latents, axis=0).mean()
    if scale_factor == 0 or np.isnan(scale_factor):
        scale_factor = 1.0
    U_scaled = U_sim / scale_factor
    V_scaled = V_sim / scale_factor
    V_mean = V_scaled.mean(axis=0)

    # ------------------------
    # Safe movie info
    # ------------------------
    movie_info_safe = movie_info_reset.copy()
    for col, default in [("movieId", list(R_full.columns)), ("title", list(R_full.columns)), ("category", "neutral")]:
        if col not in movie_info_safe.columns:
            movie_info_safe[col] = default

    # ------------------------
    # Initialize user vector & trajectory
    # ------------------------
    if st.session_state.user_latent_current is None:
        st.session_state.user_latent_current = U_scaled[user_idx, :].copy()
    uvec = st.session_state.user_latent_current
    if uvec.shape[0] < k:
        uvec = np.hstack([uvec, np.zeros(k - uvec.shape[0])])
    elif uvec.shape[0] > k:
        uvec = uvec[:k]
    st.session_state.user_latent_current = uvec
    if not st.session_state.trajectory:
        st.session_state.trajectory = [uvec.copy()]

    # ------------------------
    # Predicted ratings & top-10
    # ------------------------
    pred_ratings = st.session_state.user_latent_current @ V_scaled.T
    pred_ratings = np.ravel(pred_ratings)
    if len(pred_ratings) < n_movies:
        pad_val = np.nanmean(pred_ratings) if pred_ratings.size > 0 else 0.0
        pred_ratings = np.hstack([pred_ratings, np.full(n_movies - len(pred_ratings), pad_val)])
    else:
        pred_ratings = pred_ratings[:n_movies]

    pred_df = pd.DataFrame({"movieId": list(R_full.columns), "pred_rating": pred_ratings})
    rated_movies = []
    try:
        rated_movies = R_display.loc[selected_user].dropna().index.tolist()
    except Exception:
        pass
    pred_df = pred_df[~pred_df["movieId"].isin(rated_movies)].reset_index(drop=True)
    top10 = pred_df.sort_values("pred_rating", ascending=False).head(10)
    top10 = top10.merge(movie_info_safe[["movieId","title","category"]], on="movieId", how="left")

    # ------------------------
    # Return all necessary objects
    # ------------------------
    return {
        "U_scaled": U_scaled,
        "V_scaled": V_scaled,
        "V_mean": V_mean,
        "movie_info_safe": movie_info_safe,
        "pred_df": pred_df,
        "top10": top10,
        "k": k
    }
