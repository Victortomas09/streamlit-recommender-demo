import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objects as go  
import plotly.express as px


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
original_df = pd.read_csv("merged_movielens.csv")

merged_df = pd.read_csv("movielens_100k_categories.csv")
merged_df.columns = merged_df.columns.str.strip()

user_cols = ["userId", "age", "gender", "occupation", "zip_code"]
users = pd.read_csv("../data/ml-100k/u.user", sep="|", names=user_cols, encoding="latin-1")
users = users.drop(columns=["zip_code"])
users.columns = users.columns.str.strip()

movie_info_cols = ["movieId", "title", "category"]
movie_info = merged_df[movie_info_cols].drop_duplicates()
movie_info_reset = movie_info.reset_index(drop=True)

st.set_page_config(page_title="SVD Recommendation System", layout="wide")

# ------------------------
# Sidebar (global)
# ------------------------
st.sidebar.header("Global Controls")

# Select subset size
n_users = st.sidebar.slider("Number of users to include", 5, len(merged_df['userId'].unique()), 50)
n_movies = st.sidebar.slider("Number of movies to include", 10, len(merged_df['movieId'].unique()), 100)

# Select user and method
selected_user = st.sidebar.selectbox(
    "Select user for recommendations / simulation",
    merged_df['userId'].unique()[:n_users]
)

method = st.sidebar.selectbox(
    "Recommendation Method / Latent Model",
    ["SVD", "ALS", "PMF"]
)

# Dynamically set maximum k based on available data
max_k = min(n_users, n_movies)
latent_factors = st.sidebar.slider(
    "Latent factors (k)",
    min(3, max_k),  # Lower bound at least 3
    max_k,
    min(5, max_k)   # Default value safely within range
)

# Simulation controls
rounds = st.sidebar.slider("Simulation rounds", 1, 10, 5)
learning_rate = st.sidebar.slider("Simulation learning rate", 0.05, 0.5, 0.2)

# ------------------------
# Subset Data
# ------------------------
user_ids = merged_df['userId'].unique()[:n_users]
movie_ids = merged_df['movieId'].unique()[:n_movies]
merged_df_sub = merged_df[
    merged_df['userId'].isin(user_ids) & merged_df['movieId'].isin(movie_ids)
]
df_display = merged_df_sub.merge(users, on="userId", how="left")

# Create computational and display matrices separately
R_full = df_display.pivot(index='userId', columns='movieId', values='rating').fillna(0)  # for math
R_display = df_display.pivot(index='userId', columns='movieId', values='rating')         # for UI

als_model = None
U_pmf = None
V_pmf = None

# ------------------------
# Tabs
# ------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Recommendations", "Simulation", "Diversity Re-ranking"])

# ------------------------
# Tab 1: Data Display
# ------------------------
with tab1:
    st.header("Movielens Subset with User and Movie Info")

    st.subheader("Original Dataset ")
    st.markdown(f"**Table size:** {original_df.shape[0]} rows × {original_df.shape[1]} columns")

    st.dataframe(original_df)

    # Count selected users and movies
    n_users_selected = len(user_ids)
    n_movies_selected = len(movie_ids)

    # ------------------------
    # Ratings Table
    # ------------------------
    st.subheader(f"Ratings Table — {n_users_selected} users × {n_movies_selected} movies")

    cols_to_show = ["userId", "age", "gender", "occupation", "movieId", "title", "category", "rating"]
    existing_cols = [c for c in cols_to_show if c in df_display.columns]

    df_to_display = df_display[existing_cols]
    st.markdown(f"**Table size:** {df_to_display.shape[0]} rows × {df_to_display.shape[1]} columns")

    # Display with clean NaN handling (avoid serialization warnings)
    st.dataframe(df_to_display.style.format(na_rep=""))

    # ------------------------
    # User–Item Matrix
    # ------------------------
    st.subheader("User–Item Matrix")
    st.markdown(f"**Matrix shape:** {R_full.shape[0]} users × {R_full.shape[1]} movies")

    # Optional: hide zeros to make unrated entries look blank
    R_display = R_full.copy()
    R_display = R_display.replace(0, np.nan)
    #st.dataframe(R_display.style.format(na_rep=""))
    #st.dataframe(R_display.fillna(""))
    #st.dataframe(R_display.astype(str).replace("nan", ""))

    st.dataframe(R_display)

with tab2:
    st.header("Matrix Factorization and Latent Space Exploration")

    # ------------------------
    # Prepare rating matrix for computation
    # ------------------------
    R = df_display.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    R_matrix = R.values
    users_idx = list(R.index)
    movies_idx = list(R.columns)

    # Pivot for checking actual ratings (keep NaN for unrated movies)
    R_actual = df_display.pivot(index='userId', columns='movieId', values='rating')

    if selected_user not in users_idx:
        st.warning("Selected user is not in the current subset.")
        st.stop()

    st.subheader(f"Method: {method}")

    # ------------------------
    # Factorization
    # ------------------------
    if method == "SVD":
        U, s, Vt = np.linalg.svd(R_matrix, full_matrices=False)
        k_use = min(latent_factors, U.shape[1])
        U_k = U[:, :k_use]
        S_k = np.diag(s[:k_use])
        Vt_k = Vt[:k_use, :]
        sqrt_S = np.diag(np.sqrt(s[:k_use]))
        user_latent = U_k @ sqrt_S
        movie_latent = Vt_k.T @ sqrt_S
        pred_ratings = user_latent @ movie_latent.T
        sigma = s[:k_use]

    elif method == "ALS" and ALS_AVAILABLE:
        import scipy.sparse as sp
        sparse_R = sp.csr_matrix(R_matrix)
        als_model = AlternatingLeastSquares(
            factors=latent_factors,
            regularization=0.1,
            iterations=20
        )
        als_model.fit(sparse_R)
        user_latent = als_model.user_factors
        movie_latent = als_model.item_factors
        pred_ratings = user_latent @ movie_latent.T

    elif method == "PMF":
        lr, reg, epochs = 0.01, 0.1, 50

        np.random.seed(42)
        R = np.asarray(R_matrix, dtype=float)

        R[R == 0] = np.nan

        n_users, n_items = R.shape
        k = latent_factors

        U_pmf = np.random.normal(scale=0.1, size=(n_users, k))
        V_pmf = np.random.normal(scale=0.1, size=(n_items, k))

        # Observed entries (where R is not NaN)
        obs = np.argwhere(~np.isnan(R))

        for epoch in range(epochs):
            np.random.shuffle(obs)
            for i, j in obs:
                r = R[i, j]
                u_i = U_pmf[i].copy()
                v_j = V_pmf[j].copy()
                err = r - u_i @ v_j
                U_pmf[i] += lr * (err * v_j - reg * u_i)
                V_pmf[j] += lr * (err * u_i - reg * v_j)

        user_latent = U_pmf
        movie_latent = V_pmf

        pred_ratings = U_pmf @ V_pmf.T

    else:
        st.error("ALS not available or unknown method.")
        st.stop()

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


    # ------------------------
    # Latent Matrices Display
    # ------------------------
   
    if method == "SVD":

        col1, col2  = st.columns(2)
        with col1:

            st.latex(
            r"""
            R = U \Sigma V^{T}
            """
            )

            st.latex(
                r"""
                \hat{R} \approx U_k \Sigma_k V_k^{T}
                """
            )

            st.latex(
                r"""
                \hat{R} \approx (U_k \sqrt{\Sigma_k})(V_k \sqrt{\Sigma_k})^{T}
                """
            )

        with col2: 
            st.latex(
                r"""
                \begin{array}{l}
                \scriptsize
                \text{Where:} \\
                \scriptsize
                R \text{: original user–item rating matrix} \\
                \scriptsize
                \hat{R} \text{: reconstructed (predicted) ratings} \\
                \scriptsize
                U_k \text{: user latent feature matrix} \\
                \scriptsize
                \Sigma_k \text{: top-}k\text{ singular values (diagonal matrix)} \\
                \scriptsize
                V_k \text{: item latent feature matrix} \\
                \scriptsize
                (U_k \sqrt{\Sigma_k}) \text{: user latent factors used in prediction} \\
                \scriptsize
                (V_k \sqrt{\Sigma_k}) \text{: item latent factors used in prediction} \\
                \end{array}
                """
            )
        
        st.subheader("Latent Matrices")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("User factors (U√Σ):")
            st.markdown(f"**User latent matrix shape:** {user_latent.shape}")
            st.dataframe(pd.DataFrame(user_latent, index=users_idx))
        with col2:
            st.write("Singular values (Σ):")
            st.markdown(f"**Number of latent factors:** {latent_factors}")
            st.dataframe(pd.DataFrame(np.diag(s[:latent_factors]), columns=[f"dim{i+1}" for i in range(latent_factors)]))
        with col3:
            st.write("Item factors (V√Σ):")
            st.markdown(f"**Item latent matrix shape:** {movie_latent.shape}")
            st.dataframe(pd.DataFrame(movie_latent, index=movies_idx))

        # --- Explained Variance Equations  ---
        st.markdown("#### Explained Variance")

        col1, col2, col3 = st.columns(3)

        with col1: 

            st.latex(
                r"""
                \text{Explained Variance Ratio for component } i: \\
                \quad
                \text{EVR}_i = \frac{\sigma_i^2}{\sum_{j=1}^{n} \sigma_j^2}
                """
            )
            
        with col2: 

            st.latex(
                r"""
                \text{Cumulative Explained Variance:} \\
                \quad
                \text{CEV}_k = \sum_{i=1}^{k} \text{EVR}_i
                """
            )

        with col3: 
            st.latex(
            r"""
            \begin{array}{l}
            \scriptsize
            \text{Where:} \\
            \scriptsize
            \sigma_i \text{: singular value of component } i \\
            \scriptsize
            \text{EVR}_i \text{: proportion of total variance explained by component } i \\
            \scriptsize
            \text{CEV}_k \text{: cumulative explained variance up to } k \text{ components} \\
            \end{array}
            """
            )

        

        # --- Calculate and Display Explained Variance ---

        explained_variance_ratio = (s**2) / np.sum(s**2)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
       
        st.markdown(f"##### Total Explained Variance (Top {latent_factors} factors): {cumulative_explained_variance[latent_factors-1]:.4f}")

       
    else:
        if method == "ALS":
            st.latex(
                r"""
                \hat{R} \approx P Q^{T}
                """
            )

            st.latex(
                r"""
                \begin{array}{l}
                \scriptsize
                \text{Where:} \\
                \scriptsize
                \hat{R} \text{: reconstructed (predicted) ratings} \\
                \scriptsize
                P \text{: user latent feature matrix (users} \times \text{ latent factors)} \\
                \scriptsize
                Q \text{: item latent feature matrix (items} \times \text{ latent factors)} \\
                \scriptsize
                \text{ALS learns } P \text{ and } Q \text{ by alternatingly minimizing:} \\
                \scriptsize
                \min_{P, Q} \sum_{(u,i)\in \mathcal{K}} (r_{ui} - p_u^{T} q_i)^2 
                + \lambda ( \|p_u\|^2 + \|q_i\|^2 ) \\
                \scriptsize
                \text{Fix } Q \text{ and solve for } P, \text{ then fix } P \text{ and solve for } Q. \\
                \end{array}
                """
            )

        if method == "PMF":
            st.latex(
                r"""
                \hat{R} \approx P Q^{T}
                """
            )

            st.latex(
                r"""
                \begin{array}{l}
                \scriptsize
                \text{Where:} \\
                \scriptsize
                \hat{R} \text{: reconstructed (predicted) ratings} \\
                \scriptsize
                P \text{: user latent feature matrix (users} \times \text{ latent factors)} \\
                \scriptsize
                Q \text{: item latent feature matrix (items} \times \text{ latent factors)} \\
                \scriptsize
                \text{PMF models ratings as Gaussian-distributed:} \\
                \scriptsize
                p(R | P, Q, \sigma^2) = \prod_{(u,i)\in\mathcal{K}} 
                \mathcal{N}(r_{ui} | p_u^{T} q_i, \sigma^2) \\
                \scriptsize
                \text{with Gaussian priors on } P \text{ and } Q: \\
                \scriptsize
                p(P | \sigma_P^2) = \prod_u \mathcal{N}(p_u | 0, \sigma_P^2 I), \quad
                p(Q | \sigma_Q^2) = \prod_i \mathcal{N}(q_i | 0, \sigma_Q^2 I) \\
                \scriptsize
                \text{Learning is done by maximizing the log-posterior over } P, Q. \\
                \end{array}
                """
            )


        col1, col2 = st.columns(2)
        with col1:
            st.write("User factors:")
            st.dataframe(pd.DataFrame(user_latent, index=users_idx))
        with col2:
            st.write("Item factors:")
            st.dataframe(pd.DataFrame(movie_latent, index=movies_idx))
        

   
    st.markdown("### Reconstructed (predicted) ratings")
        
    # Convert pred_ratings to a labeled DataFrame with user_id and movie_id
    pred_df = pd.DataFrame(pred_ratings, index=users_idx, columns=movies_idx)
    pred_df.index.name = "user_id"
    pred_df.columns.name = "movie_id"

    st.dataframe(pred_df)

    st.subheader("Prediction Error Metrics")
    # --- Compute RMSE and MSE for SVD reconstruction ---

    # Create a mask of known ratings (non-NaN entries)
    valid_mask = ~np.isnan(R_matrix)

    # Compute squared errors only on observed ratings
    squared_errors = (R_matrix[valid_mask] - pred_ratings[valid_mask]) ** 2

    # Mean Squared Error
    mse = np.mean(squared_errors)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # --- Display LaTeX equations ---
    col1, col2, col3 = st.columns(3)
    with col1:

        st.latex(
            r"""
            \text{Mean Squared Error (MSE):} \\
            \text{MSE} = \frac{1}{|\mathcal{K}|} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui})^2
            """
        )
        st.markdown(f"##### Mean Squared Error (MSE): {mse:.4f}")
    
    with col2:
        st.latex(
        r"""
        \begin{array}{l}
        \scriptsize
        \text{Where:} \\
        \scriptsize
        r_{ui} \text{: actual rating of user } u \text{ for item } i \\
        \scriptsize
        \hat{r}_{ui} \text{: predicted rating of user } u \text{ for item } i \\
        \scriptsize
        \mathcal{K} \text{: set of known ratings (non-NaN entries in } R\text{)} \\
        \end{array}
        """
         )
       
    with col3: 
        st.latex(
            r"""
            \text{Root Mean Squared Error (RMSE)} \\
            \text{RMSE} = \sqrt{\text{MSE}}
            """
        )
        st.markdown(f"##### Root Mean Squared Error (RMSE) {rmse:.4f}")



    st.subheader(f"Predicted Ratings for Unrated Movies - User {selected_user}")
    st.dataframe(user_pred_sorted.head(20))


    # ------------------------
    # 3D Latent Space (User-Centered & Political Category Clusters)
    # ------------------------
    st.subheader("3D Latent Space – User-Centered ")
    if user_latent.shape[1] < 3:
        st.warning("Need at least 3 latent factors to plot 3D.")
    else:
        user_latent_norm = user_latent / np.linalg.norm(user_latent, axis=1, keepdims=True)
        movie_latent_norm = movie_latent / np.linalg.norm(movie_latent, axis=1, keepdims=True)

        # Center around user
        user_vec = user_latent_norm[user_idx, :3]
        movie_centered = movie_latent_norm[:, :3] - user_vec
        user_center = np.zeros(3)

        movie_df = pd.DataFrame(movie_centered, columns=["dim1","dim2","dim3"])
        movie_df["movieId"] = movies_idx
        movie_df = movie_df.merge(movie_info_reset, on="movieId", how="left")

        # Top-5 recommendations (unrated)
        top5_ids = user_pred_sorted.head(5).movieId.tolist()
        top5_subset = movie_df[movie_df["movieId"].isin(top5_ids)]

        category_colors = {"neutral": "green", "mildly_political": "orange", "extreme": "red"}
        fig = go.Figure()

        # Plot all movies by category
        for cat, color in category_colors.items():
            subset = movie_df[movie_df["category"]==cat]
            if not subset.empty:
                fig.add_trace(go.Scatter3d(
                    x=subset["dim1"], y=subset["dim2"], z=subset["dim3"],
                    mode='markers',
                    name=f"{cat.capitalize()}",
                    marker=dict(size=6, color=color, opacity=0.7),
                    text=subset["title"]
                ))

        # Highlight top-5 recommendations
        fig.add_trace(go.Scatter3d(
            x=top5_subset["dim1"], y=top5_subset["dim2"], z=top5_subset["dim3"],
            mode='markers+text',
            name="Top-5 Recommendations",
            marker=dict(
                size=10,
                color=[category_colors.get(cat,"gray") for cat in top5_subset["category"]],
                symbol="diamond",
                opacity=1
            ),
            text=top5_subset["title"]
        ))

        # Add user
        fig.add_trace(go.Scatter3d(
            x=[user_center[0]], y=[user_center[1]], z=[user_center[2]],
            mode='markers+text',
            name=f"User {selected_user}",
            marker=dict(size=10, color="blue", symbol="circle"),
            text=[f"User {selected_user}"]
        ))

        fig.update_layout(
            title=f"3D Latent Space – {method} (k={latent_factors})",
            legend=dict(x=0.02, y=0.98),
            scene=dict(
                xaxis_title='Latent Dim 1',
                yaxis_title='Latent Dim 2',
                zaxis_title='Latent Dim 3'
            ),
            height=750
        )
        st.plotly_chart(fig, use_container_width=True)




# ------------------------
# Tab 3: Latent Drift Simulation (fixed radicalization metric + scaled latent space)
# ------------------------
with tab3:
    st.header(f"Latent Drift Simulation for User {selected_user} ({method})")
    user_idx = list(R_full.index).index(selected_user)

    # ------------------------
    # Reset session state when sidebar changes
    # ------------------------
    reset_needed = (
        "prev_latent_factors" not in st.session_state
        or "prev_method" not in st.session_state
        or "prev_user" not in st.session_state
        or "prev_lr" not in st.session_state
        or st.session_state.prev_latent_factors != latent_factors
        or st.session_state.prev_method != method
        or st.session_state.prev_user != selected_user
        or st.session_state.prev_lr != learning_rate
    )

    if reset_needed:
        st.session_state.user_latent_current = None
        st.session_state.trajectory = []
        st.session_state.clicked_movie = None
        # reset radicalization baseline on big resets
        st.session_state.prev_rad_idx = 0.0

        st.session_state.prev_latent_factors = latent_factors
        st.session_state.prev_method = method
        st.session_state.prev_user = selected_user
        st.session_state.prev_lr = learning_rate

    # Ensure prev_rad_idx exists
    if "prev_rad_idx" not in st.session_state:
        st.session_state.prev_rad_idx = 0.0

    # ------------------------
    # Initialize latent embeddings
    # ------------------------
    if method == "SVD":
        U_sim, s_sim, Vt_sim = svd(R_full.values, full_matrices=False)
        k_sim = min(latent_factors, U_sim.shape[1])
        U_sim = U_sim[:, :k_sim]
        V_sim = Vt_sim[:k_sim, :].T
    elif method == "PMF":
        nmf_model = NMF(n_components=latent_factors, init="random", random_state=42, max_iter=200)
        W_sim = nmf_model.fit_transform(R_full.values)
        H_sim = nmf_model.components_
        U_sim, V_sim = W_sim, H_sim.T
    elif method == "ALS" and ALS_AVAILABLE:
        R_sparse = sp.csr_matrix(R_full.values)
        als_model = AlternatingLeastSquares(factors=latent_factors, iterations=20, regularization=0.1)
        als_model.fit(R_sparse.T)
        U_sim = als_model.user_factors.copy()
        V_sim = als_model.item_factors.copy()
    else:
        U_sim = np.zeros((R_full.shape[0], latent_factors))
        V_sim = np.zeros((R_full.shape[1], latent_factors))

    # ------------------------
    # Safe movie info
    # ------------------------
    movie_info_safe = movie_info_reset.copy()
    if "movieId" not in movie_info_safe.columns:
        movie_info_safe["movieId"] = list(R_full.columns)
    if "title" not in movie_info_safe.columns:
        movie_info_safe["title"] = movie_info_safe["movieId"].astype(str)
    if "category" not in movie_info_safe.columns:
        movie_info_safe["category"] = "neutral"

    # ------------------------
    # Scaling normalization (preserve clustering but make user movement visible)
    # ------------------------
    # avoid zero-division if V_sim constant
    k_dim = V_sim.shape[1]
    scale_factor = np.std(V_sim, axis=0).mean()
    if scale_factor == 0 or np.isnan(scale_factor):
        scale_factor = 1.0
    U_scaled = U_sim / scale_factor
    V_scaled = V_sim / scale_factor

    # ------------------------
    # Initialize user vector and trajectory
    # ------------------------
    if "user_latent_current" not in st.session_state or st.session_state.user_latent_current is None:
        base_vec = U_scaled[user_idx, :].copy()
        if base_vec.shape[0] < latent_factors:
            pad = np.zeros(latent_factors - base_vec.shape[0])
            base_vec = np.hstack([base_vec, pad])
        st.session_state.user_latent_current = base_vec
    if "trajectory" not in st.session_state or not st.session_state.trajectory:
        st.session_state.trajectory = [st.session_state.user_latent_current.copy()]

    current_user_vec = st.session_state.user_latent_current.copy()

    # ------------------------
    # Compute recommendations (consistent with V_scaled)
    # ------------------------
    pred_user_ratings = current_user_vec @ V_scaled.T  # (n_items,)
    try:
        rated_movies = R_display.loc[selected_user].dropna().index.tolist()
    except Exception:
        rated_movies = []

    pred_df = pd.DataFrame({
        "movieId": list(R_full.columns),
        "pred_rating": pred_user_ratings
    })
    pred_df = pred_df[~pred_df["movieId"].isin(rated_movies)].reset_index(drop=True)
    top10 = pred_df.sort_values("pred_rating", ascending=False).head(10).copy()
    top10 = top10.merge(movie_info_safe[["movieId", "title", "category"]], on="movieId", how="left")

    

    # ------------------------
    # Clickable movie buttons (single block, consistent)
    # ------------------------
    st.subheader("Click one of the top-10 recommended movies to simulate interaction")
    emoji_map = {"neutral": "🟢", "mildly_political": "🟠", "extreme": "🔴"}
    category_colors = {"neutral": "green", "mildly_political": "orange", "extreme": "red"}

    cols_row1 = st.columns(5)
    cols_row2 = st.columns(5)

    for idx, row in top10.reset_index(drop=True).iterrows():
        emo = emoji_map.get(row.get("category", "neutral"), "⚪")
        title = row.get("title", str(row["movieId"]))
        short_title = title if len(title) <= 35 else title[:32] + "..."
        label = f"{emo} {short_title}"
        key = f"sim_btn_{int(row['movieId'])}"

        container = cols_row1[idx] if idx < 5 else cols_row2[idx - 5]

        if container.button(label, key=key, help=f"Category: {row.get('category','neutral')}"):
            # update using scaled vectors (consistency)
            try:
                movie_col_index = list(R_full.columns).index(row["movieId"])
            except ValueError:
                # fallback in rare mismatch cases
                movie_col_index = 0
                for i, mid in enumerate(R_full.columns):
                    if mid == row["movieId"]:
                        movie_col_index = i
                        break

            update = V_scaled[movie_col_index, :] - st.session_state.user_latent_current
            st.session_state.user_latent_current += learning_rate * update
            st.session_state.trajectory.append(st.session_state.user_latent_current.copy())
            st.session_state.clicked_movie = row["movieId"]
            st.success(f"Simulated interaction: {title}")

    # ------------------------
    # Optional corrective action
    # ------------------------
    if st.button("🎬 Watch a Neutral Movie (Recovery Action)"):
        neutral_mask = movie_info_safe["category"] == "neutral"
        if neutral_mask.any():
            neutral_vec = np.mean(V_scaled[np.where(neutral_mask)[0], :], axis=0)
            st.session_state.user_latent_current += learning_rate * (neutral_vec - st.session_state.user_latent_current)
            st.session_state.trajectory.append(st.session_state.user_latent_current.copy())
            st.warning("You watched a neutral movie — your preferences shift back toward balance.")
        else:
            st.warning("No 'neutral' movies available in this subset to perform recovery.")

    # ------------------------
    # Visual projection & 3D plot (scaled; preserve clustering)
    # ------------------------
    st.subheader("3D Latent Space – User-Centered")

    if latent_factors < 3:
        st.warning("Need at least 3 latent factors for 3D visualization.")
    else:
        import plotly.graph_objects as go
        # Build 3D coords (pad or truncate)
        n_items, k_dim = V_scaled.shape
        if k_dim < 3:
            movie_coords = np.hstack([V_scaled, np.zeros((n_items, 3 - k_dim))])
        else:
            movie_coords = V_scaled[:, :3]

        movie_df = pd.DataFrame(movie_coords, columns=["dim1", "dim2", "dim3"])
        movie_df["movieId"] = list(R_full.columns)
        movie_df = movie_df.merge(movie_info_safe[["movieId", "title", "category"]], on="movieId", how="left")

        # Center view on current (last) user position for user-centered frame
        user_center = st.session_state.user_latent_current[:3]
        movie_df[["dim1", "dim2", "dim3"]] = movie_df[["dim1", "dim2", "dim3"]].values - user_center

        # top10 plotting subset
        top10_ids = top10["movieId"].tolist()
        top10_plot = movie_df[movie_df["movieId"].isin(top10_ids)].copy()
        # preserve order
        top10_plot["order"] = top10_plot["movieId"].map({m: i for i, m in enumerate(top10_ids)})
        top10_plot = top10_plot.sort_values("order")

        fig = go.Figure()

        # Plot movie clusters by category
        for cat, color in category_colors.items():
            subset = movie_df[movie_df["category"] == cat]
            if not subset.empty:
                fig.add_trace(go.Scatter3d(
                    x=subset["dim1"], y=subset["dim2"], z=subset["dim3"],
                    mode='markers',
                    name=f"{cat.capitalize()}",
                    marker=dict(size=5, color=color, opacity=0.6),
                    text=subset["title"]
                ))

        # Plot top-10 as diamonds
        if not top10_plot.empty:
            fig.add_trace(go.Scatter3d(
                x=top10_plot["dim1"], y=top10_plot["dim2"], z=top10_plot["dim3"],
                mode='markers+text',
                name="Top-10 (recommendations)",
                marker=dict(
                    size=9,
                    color=[category_colors.get(c, "gray") for c in top10_plot["category"]],
                    symbol="diamond",
                    opacity=1
                ),
                text=top10_plot["title"],
                textposition="top center"
            ))

        # trajectory (use scaled vectors)
        trajectory_arr = np.array(st.session_state.trajectory)
        if trajectory_arr.ndim == 1:
            trajectory_arr = trajectory_arr.reshape(1, -1)
        # pad if needed
        if trajectory_arr.shape[1] < 3:
            pad = np.zeros((trajectory_arr.shape[0], 3 - trajectory_arr.shape[1]))
            trajectory_arr = np.hstack([trajectory_arr, pad])
        traj_centered = trajectory_arr[:, :3] - user_center

        if traj_centered.shape[0] > 0:
            fig.add_trace(go.Scatter3d(
                x=traj_centered[:, 0], y=traj_centered[:, 1], z=traj_centered[:, 2],
                mode="lines+markers+text",
                name="User Trajectory",
                line=dict(color="blue", width=4),
                marker=dict(size=6, color="blue"),
                text=[f"Step {i}" for i in range(traj_centered.shape[0])],
                textposition="bottom center"
            ))

        # current user point
        cur = traj_centered[-1]
        fig.add_trace(go.Scatter3d(
            x=[cur[0]], y=[cur[1]], z=[cur[2]],
            mode="markers+text",
            name=f"User {selected_user}",
            marker=dict(size=10, color="blue", symbol="circle"),
            text=[f"User {selected_user}"],
            textposition="top center"
        ))

        fig.update_layout(
            title=f"3D Latent Space – {method} (k={latent_factors}) [User-Centered]",
            scene=dict(
                xaxis_title="Latent Dim 1",
                yaxis_title="Latent Dim 2",
                zaxis_title="Latent Dim 3"
            ),
            height=750,
            legend=dict(x=0.02, y=0.98)
        )
        st.plotly_chart(fig, use_container_width=True)


        # ------------------------
        # Radicalization Index (display before updating prev for correct delta)
        # ------------------------
        rad_idx = float((top10["category"] == "extreme").mean())
        delta_val = rad_idx - st.session_state.prev_rad_idx
        st.metric("Radicalization Index", f"{rad_idx*100:.1f}%", delta=f"{delta_val*100:+.1f}%")
        # update stored value AFTER display so next rerun has correct baseline
        st.session_state.prev_rad_idx = rad_idx

        # ------------------------
        # Category distribution chart
        # ------------------------
        import plotly.express as px
        cat_counts = top10["category"].value_counts().reindex(
            ["neutral", "mildly_political", "extreme"], fill_value=0
        )
        cat_chart = px.bar(
            x=cat_counts.index, y=cat_counts.values,
            labels={"x": "Category", "y": "Count"},
            title="Distribution of Top-10 Recommendations",
            color=cat_counts.index,
            color_discrete_map={"neutral": "green", "mildly_political": "orange", "extreme": "red"}
        )
        st.plotly_chart(cat_chart, use_container_width=True)

        
        
        # ------------------------
        # Optional: Enhanced UMAP projection – shows category clusters AND user trajectory
        # ------------------------
        try:
            from umap import UMAP

            # Fit UMAP on movie latent vectors (global content space)
            umap_3d = UMAP(n_components=3, random_state=42, metric="cosine")
            movie_coords_umap = umap_3d.fit_transform(V_scaled)

            movie_umap_df = pd.DataFrame(movie_coords_umap, columns=["dim1", "dim2", "dim3"])
            movie_umap_df["movieId"] = list(R_full.columns)
            movie_umap_df = movie_umap_df.merge(
                movie_info_safe[["movieId", "title", "category"]], on="movieId", how="left"
            )

            # --- Project user trajectory into same UMAP space ---
            trajectory_arr = np.array(st.session_state.trajectory)
            if trajectory_arr.ndim == 1:
                trajectory_arr = trajectory_arr.reshape(1, -1)
            user_umap_coords = umap_3d.transform(trajectory_arr)
            user_umap_df = pd.DataFrame(user_umap_coords, columns=["dim1", "dim2", "dim3"])
            user_umap_df["step"] = np.arange(len(user_umap_df))

            # --- Build interactive 3D visualization ---
            fig_umap = go.Figure()

            # Plot movie clusters
            for cat, color in category_colors.items():
                subset = movie_umap_df[movie_umap_df["category"] == cat]
                if not subset.empty:
                    fig_umap.add_trace(go.Scatter3d(
                        x=subset["dim1"], y=subset["dim2"], z=subset["dim3"],
                        mode="markers",
                        name=f"{cat.capitalize()} Movies",
                        marker=dict(size=5, color=color, opacity=0.6),
                        text=subset["title"]
                    ))

            # Plot user trajectory (mapped through UMAP)
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

            # Highlight current user position
            cur = user_umap_df.iloc[-1]
            fig_umap.add_trace(go.Scatter3d(
                x=[cur["dim1"]], y=[cur["dim2"]], z=[cur["dim3"]],
                mode="markers+text",
                name=f"Current User Position",
                marker=dict(size=10, color="blue", symbol="circle"),
                text=[f"User {selected_user}"],
                textposition="top center"
            ))

            # Plot top-10 recommendations (for context)
            top10_umap = movie_umap_df[movie_umap_df["movieId"].isin(top10["movieId"])].copy()
            if not top10_umap.empty:
                fig_umap.add_trace(go.Scatter3d(
                    x=top10_umap["dim1"], y=top10_umap["dim2"], z=top10_umap["dim3"],
                    mode="markers+text",
                    name="Top-10 (UMAP)",
                    marker=dict(
                        size=9,
                        color=[category_colors.get(c, "gray") for c in top10_umap["category"]],
                        symbol="diamond",
                        opacity=1
                    ),
                    text=top10_umap["title"],
                    textposition="top center"
                ))

            fig_umap.update_layout(
                title="Enhanced UMAP Projection: Content Clusters + User Drift",
                scene=dict(
                    xaxis_title="UMAP Dim 1",
                    yaxis_title="UMAP Dim 2",
                    zaxis_title="UMAP Dim 3"
                ),
                height=700,
                legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_umap, use_container_width=True)

        except Exception as e:
            st.warning(f"UMAP projection skipped: {e}")





# ------------------------
# Tab 4: Diversity Re-ranking
# ------------------------
with tab4:
    st.header("Diversity-Aware Re-ranking")
    user_idx = list(R_full.index).index(selected_user)

    # ------------------------
    # Build predicted ratings aligned to R_full columns
    # ------------------------
    if method == "SVD":
        pred_user_ratings = (U_k @ S_k)[user_idx, :] @ Vt_k
        movie_ids_for_pred = movies_idx
    elif method == "ALS" and ALS_AVAILABLE and als_model is not None:
        # ALS returns factors in the same order as sparse_R columns
        pred_ratings_full = user_latent @ movie_latent.T
        pred_user_ratings = pred_ratings_full[user_idx, :]
        movie_ids_for_pred = movies_idx  # same length as ALS columns
    elif method == "PMF" and U_pmf is not None and V_pmf is not None:
        pred_user_ratings = U_pmf[user_idx, :] @ V_pmf.T
        movie_ids_for_pred = movies_idx
    else:
        pred_user_ratings = np.zeros(R_full.shape[1])
        movie_ids_for_pred = list(R_full.columns)

    # Ensure lengths match
    assert len(pred_user_ratings) == len(movie_ids_for_pred), "Mismatch between ratings and movie IDs"

    # ------------------------
    # Build DataFrame
    # ------------------------
    rec_df = pd.DataFrame({
        "movieId": movie_ids_for_pred,
        "pred_rating": pred_user_ratings
    }).merge(movie_info_reset[["movieId", "title", "category"]], on="movieId", how="left")

    # ------------------------
    # Filter only unrated movies
    # ------------------------
    actual_ratings = R_full.loc[selected_user, movie_ids_for_pred]
    rec_df["actual_rating"] = actual_ratings.values
    rec_df = rec_df[rec_df["actual_rating"].isna()]

    if rec_df.empty:
        st.warning("User has rated all movies in the current subset.")
    else:
        diversity_lambda = st.slider("Diversity Strength (λ)", 0.0, 1.0, 0.5)
        category_counts = rec_df['category'].value_counts().to_dict()
        rec_df['diversity_score'] = rec_df['category'].apply(lambda c: 1.0 / category_counts.get(c, 1))
        rec_df['combined_score'] = (1-diversity_lambda)*rec_df['pred_rating'] + diversity_lambda*rec_df['diversity_score']

        rec_df_sorted = rec_df.sort_values('combined_score', ascending=False).head(10)
        st.subheader("Top 10 Reranked Recommendations")
        st.dataframe(rec_df_sorted[['movieId','title','category','pred_rating','diversity_score','combined_score']])
        st.info("λ controls how strongly underrepresented categories are promoted. λ=0: pure rating, λ=1: full diversity.")




