import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from helpers.helpers_tab3_trajectory import plot_category_scatter, plot_top10_markers
import numpy as np
import plotly.express as px

def plot_category_biased_knn_graph(
    movie_umap_df,
    user_umap_df,
    top10,
    category_colors,
    K=6,
    bias_strength=1.2,
    spring_scale=5.0,
    spring_k=0.3,
    seed=42,
    exaggeration=None,      # backward compatibility
    drift_strength=None,    # backward compatibility
    trajectory_scale=3.0,   # visual amplification of user steps
    jitter_scale=0.15       # small perpendicular jitter to break collinearity
):
    """
    Category-Biased KNN Graph with Alignment-Based User Drift

    trajectory_scale: multiplies the drift steps for visibility without
    altering underlying latent geometry
    jitter_scale: small random offset added perpendicular to drift to prevent collinear steps
    """
    try:
        # -------------------------------
        # 1. Prepare item data
        # -------------------------------
        X = movie_umap_df[["dim1", "dim2", "dim3"]].values
        categories = movie_umap_df["category"].values
        movie_ids = movie_umap_df["movieId"].values

        # -------------------------------
        # 2. Cosine similarity + category bias
        # -------------------------------
        S = cosine_similarity(X)
        S_biased = S.copy()
        for i in range(len(S)):
            for j in range(len(S)):
                if categories[i] == categories[j]:
                    S_biased[i, j] *= 1.6 if categories[i] == "extreme" else bias_strength
                else:
                    S_biased[i, j] *= 0.85

        # -------------------------------
        # 3. Build KNN graph
        # -------------------------------
        neighbors = np.argsort(-S_biased, axis=1)[:, :K]
        G = nx.Graph()
        for i in range(len(S)):
            for j in neighbors[i]:
                G.add_edge(i, j, weight=S_biased[i, j])

        pos = nx.spring_layout(G, dim=3, weight="weight", seed=seed, scale=spring_scale, k=spring_k)
        graph_xyz = np.array([pos[i] for i in range(len(pos))])
        movie_df = movie_umap_df.copy()
        movie_df[["dim1", "dim2", "dim3"]] = graph_xyz

        # -------------------------------
        # 4. Compute alignment with extreme content
        # -------------------------------
        user_traj = user_umap_df[["dim1", "dim2", "dim3"]].values
        extreme_mask = categories == "extreme"
        if not extreme_mask.any():
            extreme_mask = np.ones(len(categories), dtype=bool)

        extreme_dir = X[extreme_mask].mean(axis=0)
        extreme_dir /= np.linalg.norm(extreme_dir) + 1e-8

        alignment = []
        for u in user_traj:
            u_norm = u / (np.linalg.norm(u) + 1e-8)
            alignment.append(np.dot(u_norm, extreme_dir))
        alignment = np.clip(np.array(alignment), 0.0, 1.0)

        # -------------------------------
        # 5. Map alignment → graph space with trajectory scaling + lateral jitter
        # -------------------------------
        graph_center = graph_xyz.mean(axis=0)
        top10_ids = top10["movieId"].tolist()
        top10_idx = np.where(np.isin(movie_ids, top10_ids))[0]

        target_point = graph_xyz[top10_idx].mean(axis=0) if len(top10_idx) > 0 else graph_xyz[extreme_mask].mean(axis=0)
        direction = target_point - graph_center
        unit_dir = direction / (np.linalg.norm(direction) + 1e-8)

        # Helper: small random perpendicular vector
        def random_perpendicular_vector(v, scale=jitter_scale):
            rand_vec = np.random.randn(3)
            rand_vec -= rand_vec.dot(v) * v  # orthogonal component
            return rand_vec * scale

        # Step-by-step propagation with lateral jitter
        user_traj_graph = []
        prev_point = graph_center
        for a in alignment:
            step_vec = a * trajectory_scale * unit_dir
            next_point = prev_point + step_vec + random_perpendicular_vector(unit_dir)
            user_traj_graph.append(next_point)
            prev_point = next_point

        user_graph_df = pd.DataFrame(user_traj_graph, columns=["dim1", "dim2", "dim3"])
        user_graph_df["step"] = user_umap_df["step"].values

        # -------------------------------
        # 6. Plot
        # -------------------------------
        fig = go.Figure()
        plot_category_scatter(fig, movie_df, category_colors, size=6)

        if len(top10_idx) > 0:
            plot_top10_markers(
                fig,
                movie_df.iloc[top10_idx],
                category_colors,
                size=14,
                symbol="diamond",
                name="Top-10 Recommendations",
            )

        fig.add_trace(go.Scatter3d(
            x=user_graph_df["dim1"],
            y=user_graph_df["dim2"],
            z=user_graph_df["dim3"],
            mode="lines+markers+text",
            name="User Drift Toward Extreme",
            marker=dict(size=6, color="blue"),
            line=dict(width=4, color="blue"),
            text=[f"Step {s}" for s in user_graph_df["step"]],
            textposition="bottom center",
        ))

        cur = user_graph_df.iloc[-1]
        fig.add_trace(go.Scatter3d(
            x=[cur["dim1"]],
            y=[cur["dim2"]],
            z=[cur["dim3"]],
            mode="markers+text",
            name="Current User",
            marker=dict(size=12, color="blue"),
            text=["User"],
            textposition="top center",
        ))

        fig.update_layout(
            title=(
                "Category-Biased KNN Graph (Alignment-Based User Drift)<br>"
                "<sup>User motion reflects increasing alignment with extreme content</sup>"
            ),
            height=780,
            scene=dict(
                xaxis_title="Layout X",
                yaxis_title="Layout Y",
                zaxis_title="Layout Z",
            ),
            legend=dict(x=0.02, y=0.98),
        )

        return fig

    except Exception as e:
        raise RuntimeError(f"Category-biased graph generation failed: {e}")
    

def plot_user_political_extremeness_tab3(
    df_display,
    selected_user=None,
    title="User Political Extremeness vs Average Rating Mass",
    use_predicted=False
):
    """
    Scatter of users:
    - X: rating-weighted political extremeness [0,1]
    - Y: average rating mass
    - Highlights selected user

    Handles missing categories safely.
    """
    categories = ['neutral', 'mildly_political', 'extreme']

    # 1. Pivot ratings (sum per category)
    user_cat = df_display.pivot_table(
        index="userId",
        columns="category",
        values="rating",
        aggfunc="sum",
        fill_value=0
    )

    # Ensure all categories exist
    for cat in categories:
        if cat not in user_cat.columns:
            user_cat[cat] = 0

    rating_counts = df_display.groupby("userId")["rating"].count()
    user_cat["avg_rating_mass"] = user_cat.sum(axis=1) / rating_counts

    # Center ratings per user
    user_cat_centered = user_cat[categories].sub(
        user_cat[categories].mean(axis=1), axis=0
    )

    # Extremeness score relative to user's own pattern
    user_cat["extremeness"] = (
        user_cat_centered["extreme"] + 0.5 * user_cat_centered["mildly_political"]
    ).div(user_cat_centered.abs().sum(axis=1) + 1e-8)  # avoid div0

    # Scale to [0,1]
    user_cat["extremeness"] = (user_cat["extremeness"] - user_cat["extremeness"].min()) / (
        user_cat["extremeness"].max() - user_cat["extremeness"].min() + 1e-8
    )

    # 2. Scatter plot
    fig = px.scatter(
        user_cat,
        x="extremeness",
        y="avg_rating_mass",
        hover_name=user_cat.index.astype(str),
        title=title,
        labels={
            "extremeness": "Extremeness (Rating-Weighted)",
            "avg_rating_mass": "Average Rating"
        }
    )

    fig.update_traces(marker=dict(size=7, opacity=0.65), showlegend=False)

    # 3. X-axis segmentation
    fig.update_xaxes(
        range=[0, 1],
        tickmode="array",
        tickvals=[0, 0.33, 0.66, 1.0],
        ticktext=["0", "Neutral", "Mild", "Extreme"]
    )
    fig.add_vrect(x0=0.0, x1=0.33, fillcolor="green", opacity=0.06, line_width=0)
    fig.add_vrect(x0=0.33, x1=0.66, fillcolor="orange", opacity=0.06, line_width=0)
    fig.add_vrect(x0=0.66, x1=1.0, fillcolor="red", opacity=0.06, line_width=0)
    fig.add_vline(x=0.33, line_dash="dash", line_color="orange")
    fig.add_vline(x=0.66, line_dash="dash", line_color="red")

    # 4. Highlight selected user
    if selected_user is not None and selected_user in user_cat.index:
        u = user_cat.loc[selected_user]
        fig.add_trace(go.Scatter(
            x=[u["extremeness"]],
            y=[u["avg_rating_mass"]],
            mode="markers+text",
            name=f"User {selected_user}",
            marker=dict(
                size=14,
                color="blue",
                symbol="star",
                line=dict(width=1, color="red")
            ),
            text=[f"User {selected_user}"],
            textposition="top center"
        ))

    return fig
