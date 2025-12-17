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
st.sidebar.header("Global Parameters:")

# Select subset size
n_users = st.sidebar.slider(
    "Number of users to include",
    5, len(merged_df['userId'].unique()), 50
)
n_movies = st.sidebar.slider(
    "Number of movies to include",
    10, len(merged_df['movieId'].unique()), 100
)

# Select user and method
selected_user = st.sidebar.selectbox(
    "Select user for recommendations / simulation",
    merged_df['userId'].unique()[:n_users]
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
rounds = st.sidebar.slider("Simulation rounds", 1, 10, 5)
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
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Recommendations", "Simulation", "Diversity Re-ranking"])

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

        # Mapping table
        mapping_data = {
            "Movielens Genre": [
                "Comedy", "Animation", "Children's", "Musical",
                "Action", "Adventure", "Thriller", "Crime",
                "War", "Film-Noir", "Horror",
                "Drama", "Romance", "Sci-Fi", "Mystery",
                "Fantasy", "Documentary",
                "Action-Adventure/Other", "Unknown"
            ],
            "Political Category": [
                "neutral", "neutral", "neutral", "neutral",
                "mildly_political", "mildly_political", "mildly_political", "mildly_political",
                "extreme", "extreme", "extreme",
                "neutral", "neutral", "neutral", "mildly_political",
                "neutral", "neutral",
                "mildly_political", "neutral"
            ],
            "Notes / Reasoning": [
                "Lighthearted, non-political",
                "Kid-friendly, neutral themes",
                "Family content, neutral",
                "Entertainment-focused",
                "Often aggressive, conflict-oriented",
                "Excitement, quests, some thematic tension",
                "Suspenseful or intense content",
                "Often deals with moral or social issues",
                "Conflict, violence, ideological narratives",
                "Dark, morally ambiguous, intense themes",
                "Fear-inducing, extreme reactions",
                "Everyday situations; could also be split depending on theme",
                "Love stories, neutral",
                "Speculative, usually non-political",
                "Intrigue, tension, societal questions",
                "Imaginary worlds, escapism",
                "Informational, generally neutral",
                "If ambiguous, treat as mildly political",
                "Default neutral"
            ]
        }

        category_mapping_df = pd.DataFrame(mapping_data)

        st.dataframe(category_mapping_df)


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

        # Color coding
        color_map = {
            "neutral": "green",
            "mildly_political": "orange",
            "extreme": "red"
        }

        # Compute counts
        cat_counts_full = merged_df['category'].value_counts().reset_index()
        cat_counts_full.columns = ['category', 'count']

        cat_counts_filtered = df_to_display['category'].value_counts().reset_index()
        cat_counts_filtered.columns = ['category', 'count']

        # Small side-by-side columns
        c1, c2 = st.columns([1, 1])

        with c1:
            fig_full = px.bar(
                cat_counts_full,
                x='category',
                y='count',
                title="Original Dataset",
                color='category',
                color_discrete_map=color_map,
                height=280,
                width=280
            )

            fig_full.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig_full, use_container_width=True)

        with c2:
            fig_filtered = px.bar(
                cat_counts_filtered,
                x='category',
                y='count',
                title="After Filtering",
                color='category',
                color_discrete_map=color_map,
                height=280,
                width=280
            )

            fig_filtered.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig_filtered, use_container_width=True)


    # User–Item Matrix
    st.subheader("User–Item Matrix")
    st.markdown(f"**Matrix shape:** {R_full.shape[0]} users × {R_full.shape[1]} movies")

    R_display_clean = R_full.replace(0, np.nan)
    st.dataframe(R_display_clean)


# -------------------------------------------------------
# Tab 1: Matrix Factorization and Latent Space Exploration
# --------------------------------------------------------
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
    # Latent Matrices Display
    # ------------------------
   
    if method == "SVD":

        st.subheader(f"Method: Singular Value Decomposition - SVD")

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

            st.subheader(f"Method: Alternating Least Squares - ALS")
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
            
            st.subheader(f"Method: Probabilistic Matrix Factorization - PMF")
            
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
        st.dataframe(user_pred_sorted.head(20))

    with col2:
        st.subheader("Category Distribution (Top 20 Recommendations)")

        # Color coding
        color_map = {
            "neutral": "green",
            "mildly_political": "orange",
            "extreme": "red"
        }

        # Take the top 20 recommended movies
        top20 = user_pred_sorted.head(20)

        # Count categories in top 20
        cat_counts_top20 = (
            top20["category"]
            .value_counts()
            .reset_index()
        )
        cat_counts_top20.columns = ["category", "count"]

        # Build bar chart
        fig_top20 = px.bar(
            cat_counts_top20,
            x="category",
            y="count",
            title=f"User {selected_user} Top 20 Recommendations  ",
            color="category",
            color_discrete_map=color_map,
            height=300,
            width=400
        )

        fig_top20.update_layout(
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig_top20, use_container_width=True)



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
# Tab 3: Latent Drift Simulation (robust, ALS+UMAP fixes, dynamic movie count)
# ------------------------
with tab3:
    st.header(f"Latent Drift Simulation for User {selected_user} ({method})")

    # number of movies currently present (the sidebar may dynamically change the R_full columns)
    n_movies = len(R_full.columns)
    k = latent_factors
    user_idx = list(R_full.index).index(selected_user)

    # ------------------------
    # Session-state reset logic (when sidebar changes)
    # ------------------------
    reset_needed = (
        "prev_latent_factors" not in st.session_state
        or "prev_method" not in st.session_state
        or "prev_user" not in st.session_state
        or "prev_lr" not in st.session_state
        or st.session_state.prev_latent_factors != k
        or st.session_state.prev_method != method
        or st.session_state.prev_user != selected_user
        or st.session_state.prev_lr != learning_rate
        or "prev_n_movies" not in st.session_state
        or st.session_state.prev_n_movies != n_movies
    )

    if reset_needed:
        st.session_state.user_latent_current = None
        st.session_state.clicked_movie = None
        st.session_state.prev_rad_idx = 0.0
        # reset trajectory when user/method/k/n_movies change (keeps it simple & safe)
        st.session_state.trajectory = []
        # store prevs
        st.session_state.prev_latent_factors = k
        st.session_state.prev_method = method
        st.session_state.prev_user = selected_user
        st.session_state.prev_lr = learning_rate
        st.session_state.prev_n_movies = n_movies

    if "prev_rad_idx" not in st.session_state:
        st.session_state.prev_rad_idx = 0.0

    # ------------------------
    # Compute latent factors (SVD / PMF / ALS). Make sure V_sim has shape (n_movies, k)
    # ------------------------
    if method == "SVD":
        U_sim, s_sim, Vt_sim = svd(R_full.values, full_matrices=False)
        # take min dims if needed
        k_sim = min(k, U_sim.shape[1])
        U_sim = U_sim[:, :k_sim]
        V_sim = Vt_sim[:k_sim, :].T  # shape: items × k_sim
        # pad to k if necessary
        if V_sim.shape[1] < k:
            V_sim = np.hstack([V_sim, np.zeros((V_sim.shape[0], k - V_sim.shape[1]))])
        if U_sim.shape[1] < k:
            U_sim = np.hstack([U_sim, np.zeros((U_sim.shape[0], k - U_sim.shape[1]))])

    elif method == "PMF":
        nmf_model = NMF(n_components=k, init="random", random_state=42, max_iter=200)
        W_sim = nmf_model.fit_transform(R_full.values)  # users × k
        H_sim = nmf_model.components_                   # k × items
        U_sim = W_sim
        V_sim = H_sim.T  # items × k

    elif method == "ALS" and ALS_AVAILABLE:
        # Fit ALS on the same R_full (sparse)
        R_sparse = sp.csr_matrix(R_full.values)
        als_model = AlternatingLeastSquares(factors=k, iterations=20, regularization=0.1)
        # implicit ALS expects item-user matrix when fit(R_sparse.T) was used previously;
        # using the same approach so model.item_factors corresponds to items used by the fit
        als_model.fit(R_sparse.T)

        # Note: implicit's attributes are user_factors and item_factors. We'll copy them and then
        # robustly align to the full movie list length (n_movies).
        # Usually: user_factors -> users × k, item_factors -> items × k
        try:
            U_sim_raw = als_model.user_factors.copy()   # users × k
            V_sim_raw = als_model.item_factors.copy()   # items × k
        except Exception:
            # fallback in case naming differs
            U_sim_raw = getattr(als_model, "user_factors", np.zeros((R_full.shape[0], k))).copy()
            V_sim_raw = getattr(als_model, "item_factors", np.zeros((0, k))).copy()

        # Build V_sim of shape (n_movies, k). If ALS returned fewer item rows (e.g. items with no data),
        # fill the remaining rows with the mean item vector to preserve global scale and include all movies.
        V_sim = np.zeros((n_movies, k))
        n_items_present = V_sim_raw.shape[0]
        if n_items_present > 0:
            n_copy = min(n_items_present, n_movies)
            V_sim[:n_copy, :] = V_sim_raw[:n_copy, :]

            if n_items_present < n_movies:
                mean_vec = V_sim_raw.mean(axis=0)
                V_sim[n_items_present:, :] = mean_vec
        else:
            # no item factors returned -> zeros
            V_sim[:] = 0.0

        # U_sim: ensure it has shape (n_users, k)
        U_sim = U_sim_raw.copy()
        if U_sim.shape[0] < R_full.shape[0]:
            # pad users with mean user vector
            if U_sim.shape[0] > 0:
                mean_user = U_sim.mean(axis=0)
            else:
                mean_user = np.zeros(k)
            pad_users = np.tile(mean_user, (R_full.shape[0] - U_sim.shape[0], 1))
            U_sim = np.vstack([U_sim, pad_users])
        elif U_sim.shape[0] > R_full.shape[0]:
            U_sim = U_sim[: R_full.shape[0], :]

    else:
        # fallback zeros
        U_sim = np.zeros((R_full.shape[0], k))
        V_sim = np.zeros((n_movies, k))

    # Ensure V_sim shape is exactly (n_movies, k)
    if V_sim.shape != (n_movies, k):
        V_tmp = np.zeros((n_movies, k))
        r = min(V_sim.shape[0], n_movies)
        c = min(V_sim.shape[1], k)
        V_tmp[:r, :c] = V_sim[:r, :c]
        V_sim = V_tmp

    # Ensure U_sim shape is (n_users, k)
    if U_sim.shape[1] != k:
        # pad/truncate columns
        U_tmp = np.zeros((U_sim.shape[0], k))
        c = min(U_sim.shape[1], k)
        U_tmp[:, :c] = U_sim[:, :c]
        U_sim = U_tmp
    if U_sim.shape[0] != R_full.shape[0]:
        # pad/truncate rows
        U_tmp = np.zeros((R_full.shape[0], k))
        r = min(U_sim.shape[0], R_full.shape[0])
        U_tmp[:r, :] = U_sim[:r, :]
        U_sim = U_tmp

    # ------------------------
    # Scale / normalize latent space for visualization
    # ------------------------
    scale_factor = np.std(V_sim, axis=0).mean()
    if scale_factor == 0 or np.isnan(scale_factor):
        scale_factor = 1.0
    U_scaled = U_sim / scale_factor
    V_scaled = V_sim / scale_factor

    # movie mean (used for padding trajectory / centering)
    V_mean = V_scaled.mean(axis=0)

    # ------------------------
    # Safe movie info (fill defaults if missing)
    # ------------------------
    movie_info_safe = movie_info_reset.copy()
    if "movieId" not in movie_info_safe.columns:
        movie_info_safe["movieId"] = list(R_full.columns)
    if "title" not in movie_info_safe.columns:
        movie_info_safe["title"] = movie_info_safe["movieId"].astype(str)
    if "category" not in movie_info_safe.columns:
        movie_info_safe["category"] = "neutral"

    # ------------------------
    # Initialize current user vector and trajectory (pad/truncate to k)
    # ------------------------
    if "user_latent_current" not in st.session_state or st.session_state.user_latent_current is None:
        st.session_state.user_latent_current = U_scaled[user_idx, :].copy()

    # ensure the stored user vector length matches k
    uvec = np.array(st.session_state.user_latent_current)
    if uvec.shape[0] < k:
        uvec = np.hstack([uvec, np.zeros(k - uvec.shape[0])])
    elif uvec.shape[0] > k:
        uvec = uvec[:k]
    st.session_state.user_latent_current = uvec.copy()

    if "trajectory" not in st.session_state or not st.session_state.trajectory:
        st.session_state.trajectory = [st.session_state.user_latent_current.copy()]

    current_user_vec = st.session_state.user_latent_current.copy()

    # ------------------------
    # Safe predictions (length-aligned)
    # ------------------------
    pred_user_ratings = current_user_vec @ V_scaled.T
    pred_user_ratings = np.ravel(pred_user_ratings)
    # ensure lengths match
    all_movie_ids = list(R_full.columns)
    if len(pred_user_ratings) != len(all_movie_ids):
        # pad with mean rating (or zeros) or truncate as needed
        if len(pred_user_ratings) < len(all_movie_ids):
            pad_val = np.nanmean(pred_user_ratings) if pred_user_ratings.size > 0 else 0.0
            pred_user_ratings = np.hstack([pred_user_ratings, np.full(len(all_movie_ids) - len(pred_user_ratings), pad_val)])
        else:
            pred_user_ratings = pred_user_ratings[: len(all_movie_ids)]

    # build DataFrame
    pred_df = pd.DataFrame({"movieId": all_movie_ids, "pred_rating": pred_user_ratings})

    # filter out already rated (safe)
    rated_movies = []
    try:
        rated_movies = R_display.loc[selected_user].dropna().index.tolist()
    except Exception:
        pass
    pred_df = pred_df[~pred_df["movieId"].isin(rated_movies)].reset_index(drop=True)

    top10 = pred_df.sort_values("pred_rating", ascending=False).head(10)
    top10 = top10.merge(movie_info_safe[["movieId", "title", "category"]], on="movieId", how="left")

    # ------------------------
    # Clickable movie buttons (simulate interaction)
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
            movie_col_index = all_movie_ids.index(row["movieId"])
            update = V_scaled[movie_col_index, :] - st.session_state.user_latent_current
            st.session_state.user_latent_current += learning_rate * update
            # ensure stored vector length k
            u = st.session_state.user_latent_current
            if len(u) < k:
                u = np.hstack([u, np.zeros(k - len(u))])
            elif len(u) > k:
                u = u[:k]
            st.session_state.user_latent_current = u
            st.session_state.trajectory.append(st.session_state.user_latent_current.copy())
            st.session_state.clicked_movie = row["movieId"]
            st.success(f"Simulated interaction: {title}")

    # ------------------------
    # Recovery action (neutral movie)
    # ------------------------
    if st.button("🎬 Watch a Neutral Movie (Recovery Action)"):
        neutral_mask = movie_info_safe["category"] == "neutral"
        if neutral_mask.any():
            neutral_indices = np.where(neutral_mask.values)[0]
            if neutral_indices.size > 0:
                neutral_vec = V_scaled[neutral_indices, :].mean(axis=0)
            else:
                neutral_vec = V_mean
            st.session_state.user_latent_current += learning_rate * (neutral_vec - st.session_state.user_latent_current)
            # ensure length k
            u = st.session_state.user_latent_current
            if len(u) < k:
                u = np.hstack([u, np.zeros(k - len(u))])
            elif len(u) > k:
                u = u[:k]
            st.session_state.user_latent_current = u
            st.session_state.trajectory.append(st.session_state.user_latent_current.copy())
            st.warning("You watched a neutral movie — your preferences shift back toward balance.")
        else:
            st.warning("No 'neutral' movies available in this subset to perform recovery.")

    col1,col2 = st.columns(2)

    with col1: 

        # ------------------------
        # 3D Latent Space plot (user-centered)
        # ------------------------
        st.subheader("3D Latent Space – User-Centered")
        if k < 3:
            st.warning("Need at least 3 latent factors for 3D visualization.")
        else:
            movie_coords = V_scaled[:, :3] if k >= 3 else np.hstack([V_scaled, np.zeros((n_movies, 3 - k))])
            movie_df = pd.DataFrame(movie_coords, columns=["dim1", "dim2", "dim3"])
            movie_df["movieId"] = all_movie_ids
            movie_df = movie_df.merge(movie_info_safe[["movieId", "title", "category"]], on="movieId", how="left")

            # center on current user position
            user_center = st.session_state.user_latent_current[:3]
            movie_df[["dim1", "dim2", "dim3"]] = movie_df[["dim1", "dim2", "dim3"]].values - user_center

            # top10 subset (preserve order)
            top10_ids = top10["movieId"].tolist()
            top10_plot = movie_df[movie_df["movieId"].isin(top10_ids)].copy()
            top10_plot["order"] = top10_plot["movieId"].map({m: i for i, m in enumerate(top10_ids)})
            top10_plot = top10_plot.sort_values("order")

            fig = go.Figure()
            # clusters by category
            for cat, color in category_colors.items():
                subset = movie_df[movie_df["category"] == cat]
                if not subset.empty:
                    fig.add_trace(go.Scatter3d(
                        x=subset["dim1"], y=subset["dim2"], z=subset["dim3"],
                        mode='markers', name=f"{cat.capitalize()}",
                        marker=dict(size=5, color=color, opacity=0.6),
                        text=subset["title"]
                    ))

            # top10 diamonds
            if not top10_plot.empty:
                fig.add_trace(go.Scatter3d(
                    x=top10_plot["dim1"], y=top10_plot["dim2"], z=top10_plot["dim3"],
                    mode='markers+text', name="Top-10 (recommendations)",
                    marker=dict(size=9, color=[category_colors.get(c, "gray") for c in top10_plot["category"]],
                                symbol="diamond", opacity=1),
                    text=top10_plot["title"], textposition="top center"
                ))

            # trajectory (pad/truncate to k, then take first 3 dims and center by user_center)
            trajectory_arr = np.array(st.session_state.trajectory)
            if trajectory_arr.ndim == 1:
                trajectory_arr = trajectory_arr.reshape(1, -1)
            # pad/truncate columns to k
            traj_k = trajectory_arr.shape[1]
            if traj_k < k:
                trajectory_arr = np.hstack([trajectory_arr, np.tile(V_mean, (trajectory_arr.shape[0], k - traj_k))])
            elif traj_k > k:
                trajectory_arr = trajectory_arr[:, :k]
            traj3 = trajectory_arr[:, :3] - user_center
            if traj3.shape[0] > 0:
                fig.add_trace(go.Scatter3d(
                    x=traj3[:, 0], y=traj3[:, 1], z=traj3[:, 2],
                    mode="lines+markers+text", name="User Trajectory",
                    line=dict(color="blue", width=4),
                    marker=dict(size=6, color="blue"),
                    text=[f"Step {i}" for i in range(traj3.shape[0])],
                    textposition="bottom center"
                ))

            # current user
            cur = traj3[-1]
            fig.add_trace(go.Scatter3d(
                x=[cur[0]], y=[cur[1]], z=[cur[2]],
                mode="markers+text", name=f"User {selected_user}",
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

    with col2: 
        # ------------------------
        # Enhanced UMAP projection (fit once per method,n_movies,k and reuse)
        # ------------------------
        try:
            umap_key = f"umap_{method}_{n_movies}_{k}"
            # We'll store model and V_mean (for padding) in session_state so transforms remain consistent
            if umap_key not in st.session_state:
                # build input for UMAP: center by V_mean then normalize rows to unit length
                V_center = V_scaled - V_mean
                row_norms = np.linalg.norm(V_center, axis=1, keepdims=True)
                V_umap_input = V_center / (row_norms + 1e-9)

                umap_model = UMAP(n_components=3, random_state=42, metric="cosine")
                movie_coords_umap = umap_model.fit_transform(V_umap_input)

                st.session_state[umap_key] = {
                    "model": umap_model,
                    "V_mean": V_mean,
                    "row_norms": row_norms  # helpful if needed for consistent transform
                }
            else:
                umap_model = st.session_state[umap_key]["model"]
                # ensure V_mean used when padding trajectory
                V_mean_stored = st.session_state[umap_key].get("V_mean", V_mean)

                # prepare V_umap_input same way (center & normalize)
                V_center = V_scaled - V_mean_stored
                row_norms = np.linalg.norm(V_center, axis=1, keepdims=True)
                V_umap_input = V_center / (row_norms + 1e-9)
                movie_coords_umap = umap_model.transform(V_umap_input)

            # movie UMAP dataframe (movie order aligned to R_full.columns)
            movie_umap_df = pd.DataFrame(movie_coords_umap, columns=["dim1", "dim2", "dim3"])
            movie_umap_df["movieId"] = all_movie_ids
            movie_umap_df = movie_umap_df.merge(movie_info_safe[["movieId", "title", "category"]], on="movieId", how="left")

            # Transform entire stored trajectory consistently: center by same V_mean then normalize rows
            trajectory_arr = np.array(st.session_state.trajectory)
            if trajectory_arr.ndim == 1:
                trajectory_arr = trajectory_arr.reshape(1, -1)
            # pad/truncate to k
            traj_k = trajectory_arr.shape[1]
            if traj_k < k:
                trajectory_arr = np.hstack([trajectory_arr, np.tile(V_mean, (trajectory_arr.shape[0], k - traj_k))])
            elif traj_k > k:
                trajectory_arr = trajectory_arr[:, :k]

            # center using stored V_mean (if available)
            V_mean_for_traj = st.session_state[umap_key].get("V_mean", V_mean)
            traj_center = trajectory_arr - V_mean_for_traj
            traj_norms = np.linalg.norm(traj_center, axis=1, keepdims=True)
            traj_umap_input = traj_center / (traj_norms + 1e-9)

            user_umap_coords = st.session_state[umap_key]["model"].transform(traj_umap_input)
            user_umap_df = pd.DataFrame(user_umap_coords, columns=["dim1", "dim2", "dim3"])
            user_umap_df["step"] = np.arange(len(user_umap_df))

            # plotting UMAP
            fig_umap = go.Figure()
            for cat, color in category_colors.items():
                subset = movie_umap_df[movie_umap_df["category"] == cat]
                if not subset.empty:
                    fig_umap.add_trace(go.Scatter3d(
                        x=subset["dim1"], y=subset["dim2"], z=subset["dim3"],
                        mode='markers', name=f"{cat.capitalize()} Movies",
                        marker=dict(size=5, color=color, opacity=0.6),
                        text=subset["title"]
                    ))

            if len(user_umap_df) > 1:
                fig_umap.add_trace(go.Scatter3d(
                    x=user_umap_df["dim1"], y=user_umap_df["dim2"], z=user_umap_df["dim3"],
                    mode="lines+markers+text", name="User Trajectory (UMAP)",
                    line=dict(color="blue", width=4),
                    marker=dict(size=6, color="blue"),
                    text=[f"Step {i}" for i in user_umap_df["step"]],
                    textposition="bottom center"
                ))

            # current user
            cur = user_umap_df.iloc[-1]
            fig_umap.add_trace(go.Scatter3d(
                x=[cur["dim1"]], y=[cur["dim2"]], z=[cur["dim3"]],
                mode="markers+text", name="Current User Position",
                marker=dict(size=10, color="blue", symbol="circle"),
                text=[f"User {selected_user}"], textposition="top center"
            ))

            # top10 context
            top10_umap = movie_umap_df[movie_umap_df["movieId"].isin(top10["movieId"])].copy()
            if not top10_umap.empty:
                fig_umap.add_trace(go.Scatter3d(
                    x=top10_umap["dim1"], y=top10_umap["dim2"], z=top10_umap["dim3"],
                    mode="markers+text", name="Top-10 (UMAP)",
                    marker=dict(size=9, color=[category_colors.get(c, "gray") for c in top10_umap["category"]],
                                symbol="diamond", opacity=1),
                    text=top10_umap["title"], textposition="top center"
                ))

            fig_umap.update_layout(
                title="Enhanced UMAP Projection (Cosine distance): Content Clusters + User Drift",
                scene=dict(xaxis_title="UMAP Dim 1", yaxis_title="UMAP Dim 2", zaxis_title="UMAP Dim 3"),
                height=750, legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_umap, use_container_width=True)

        except Exception as e:
            st.warning(f"UMAP projection skipped: {e}")



    # ------------------------
    # Radicalization Index (top10-based)
    # ------------------------
    rad_idx = float((top10["category"] == "extreme").mean())
    delta_val = rad_idx - st.session_state.prev_rad_idx
    st.metric("Radicalization Index", f"{rad_idx*100:.1f}%", delta=f"{delta_val*100:+.1f}%")
    st.session_state.prev_rad_idx = rad_idx

    # ------------------------
    # Category distribution chart
    # ------------------------
    cat_counts = top10["category"].value_counts().reindex(["neutral", "mildly_political", "extreme"], fill_value=0)
    cat_chart = px.bar(
        x=cat_counts.index, y=cat_counts.values,
        labels={"x": "Category", "y": "Count"},
        title="Distribution of Top-10 Recommendations",
        color=cat_counts.index,
        color_discrete_map={"neutral": "green", "mildly_political": "orange", "extreme": "red"}
    )
    st.plotly_chart(cat_chart, use_container_width=True)

    

    import numpy as np
    import numpy as np
    import networkx as nx
    import plotly.graph_objects as go
    from sklearn.metrics.pairwise import cosine_similarity

    st.subheader("Category-Biased KNN Graph (Top-10 Highlight + Exaggerated User Drift)")

    try:
        # -------------------------------
        # 1. Prepare UMAP coordinates
        # -------------------------------
        X = movie_umap_df[["dim1", "dim2", "dim3"]].values
        categories = movie_umap_df["category"].values
        titles = movie_umap_df["title"].values
        movie_ids = movie_umap_df["movieId"].values

        # -------------------------------
        # 2. Compute cosine similarity
        # -------------------------------
        S = cosine_similarity(X)

        # -------------------------------
        # 3. Apply category bias (visual only)
        # -------------------------------
        bias_strength = 1.2
        S_biased = S.copy()
        for i in range(len(S)):
            for j in range(len(S)):
                if categories[i] == categories[j]:
                    S_biased[i, j] *= bias_strength

        # -------------------------------
        # 4. Build KNN graph
        # -------------------------------
        K = 6
        neighbors = np.argsort(-S_biased, axis=1)[:, :K]

        G = nx.Graph()
        for i in range(len(S)):
            for j in neighbors[i]:
                G.add_edge(i, j, weight=S_biased[i, j])

        # -------------------------------
        # 5. Graph layout (3D Spring)
        # -------------------------------
        pos = nx.spring_layout(G, dim=3, weight="weight", seed=42, scale=5.0, k=0.3)
        graph_xyz = np.array([pos[i] for i in range(len(pos))])

        # -------------------------------
        # 6. Safe trajectory exaggeration toward clicked movies/clusters
        # -------------------------------
        user_traj = user_umap_df[["dim1","dim2","dim3"]].values
        traj_start = user_traj[0]

        # Determine target location: last Top-10 movie centroid
        top10_ids = top10["movieId"].tolist()
        top10_idx = np.where(np.isin(movie_ids, top10_ids))[0]
        if len(top10_idx) > 0:
            target_point = graph_xyz[top10_idx].mean(axis=0)
        else:
            target_point = graph_xyz.mean(axis=0)

        # 6a. Compute trajectory offsets and exaggerate
        trajectory_exaggeration = 1.1  # moderate
        traj_offsets = user_traj - traj_start
        user_traj_scaled = traj_start + traj_offsets * trajectory_exaggeration

        # -------------------------------
        # 6b. Moderate directional drift toward target
        # -------------------------------
        direction_to_target = target_point - user_traj_scaled[-1]
        drift_strength = 0.3  # small factor to move toward clicked movies/clusters
        user_traj_scaled += np.linspace(0, 1, len(user_traj_scaled)).reshape(-1,1) * direction_to_target * drift_strength
        
        # 6c. Center trajectory near cluster center (optional)
        cluster_center = graph_xyz.mean(axis=0)
        user_traj_scaled += cluster_center - traj_start

        # 6d. Put into DataFrame
        user_umap_df_scaled = pd.DataFrame(user_traj_scaled, columns=["dim1","dim2","dim3"])
        user_umap_df_scaled["step"] = user_umap_df["step"].values

        # -------------------------------
        # 7. Plot: Category-Biased Layout
        # -------------------------------
        fig_knn = go.Figure()

        # plot movies by category
        for cat, color in category_colors.items():
            idx = np.where(categories == cat)[0]
            if len(idx) > 0:
                fig_knn.add_trace(go.Scatter3d(
                    x=graph_xyz[idx, 0],
                    y=graph_xyz[idx, 1],
                    z=graph_xyz[idx, 2],
                    mode="markers",
                    name=f"{cat.capitalize()}",
                    marker=dict(size=6, opacity=0.75, color=color),
                    text=titles[idx]
                ))

        # -------------------------------
        # 8. Highlight Top-10 Recommendations
        # -------------------------------
        if len(top10_idx) > 0:
            fig_knn.add_trace(go.Scatter3d(
                x=graph_xyz[top10_idx, 0],
                y=graph_xyz[top10_idx, 1],
                z=graph_xyz[top10_idx, 2],
                mode="markers+text",
                name="Top-10 Recommendations",
                marker=dict(
                    size=14,
                    symbol="diamond",
                    opacity=1,
                    color=[category_colors.get(categories[i], "gray") for i in top10_idx]
                ),
                text=[titles[i] for i in top10_idx],
                textposition="top center"
            ))

        # -------------------------------
        # 9. Overlay Exaggerated User Trajectory
        # -------------------------------
        if len(user_umap_df_scaled) > 1:
            fig_knn.add_trace(go.Scatter3d(
                x=user_umap_df_scaled["dim1"],
                y=user_umap_df_scaled["dim2"],
                z=user_umap_df_scaled["dim3"],
                mode="lines+markers+text",
                name="User Trajectory (Exaggerated)",
                marker=dict(size=6, color="black"),
                line=dict(width=4, color="black"),
                text=[f"Step {s}" for s in user_umap_df_scaled["step"]],
                textposition="bottom center"
            ))

        # current user
        cur = user_umap_df_scaled.iloc[-1]
        fig_knn.add_trace(go.Scatter3d(
            x=[cur["dim1"]],
            y=[cur["dim2"]],
            z=[cur["dim3"]],
            mode="markers+text",
            name="Current User",
            marker=dict(size=12, color="black", symbol="circle"),
            text=["User"],
            textposition="top center"
        ))

        # -------------------------------
        # 10. Layout
        # -------------------------------
        fig_knn.update_layout(
            title=(
                "Category-Biased KNN Graph (Visualization Only)<br>"
                "<sup>Clusters are visually enhanced — MF/UMAP embeddings unchanged. "
                "User trajectory exaggerated toward clicked movies for demo clarity.</sup>"
            ),
            height=780,
            scene=dict(
                xaxis_title="Layout X",
                yaxis_title="Layout Y",
                zaxis_title="Layout Z"
            ),
            legend=dict(x=0.02, y=0.98),
        )

        st.plotly_chart(fig_knn, use_container_width=True)

    except Exception as e:
        st.warning(f"Category-biased graph skipped: {e}")









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




