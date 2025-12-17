import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



def get_category_mapping_df() -> pd.DataFrame:
    """Return the genre-to-political-category mapping as a DataFrame."""
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
            "mildly_political", "mildly_political", "mildly_political", "extreme",
            "extreme", "extreme", "extreme",
            "extreme", "neutral", "neutral", "mildly_political",
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
            "If ambiguous, treat as mildly_political",
            "Default neutral"
        ]
    }
    return pd.DataFrame(mapping_data)


def plot_category_distribution(cat_counts_full, cat_counts_filtered, color_map):
    """Render the two bar charts (full vs filtered) side-by-side."""
    c1, c2 = st.columns(2)

    with c1:
        fig_full = px.bar(
            cat_counts_full, x="category", y="count",
            title="Original Dataset",
            color="category", color_discrete_map=color_map,
            height=280, width=280
        )
        fig_full.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_full, use_container_width=True)

    with c2:
        fig_filtered = px.bar(
            cat_counts_filtered, x="category", y="count",
            title="After Filtering",
            color="category", color_discrete_map=color_map,
            height=280, width=280
        )
        fig_filtered.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_filtered, use_container_width=True)






def plot_user_political_extremeness(
    df_display,
    selected_user=None,
    title="User Political Extremeness vs Average Rating Mass",
    use_predicted=False
):
    """
    Plots users in 2D:
    - X: rating-weighted political extremeness in [0,1]
    - Y: average rating mass
    - Highlights a selected user if provided

    Parameters
    ----------
    df_display : pd.DataFrame
        Must contain columns: userId, category, rating
        If use_predicted=True, 'rating' should be predicted rating
    selected_user : optional
        userId to highlight
    title : str
        Plot title
    use_predicted : bool
        Whether the ratings are predicted (SVD/ALS/PMF) or actual

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    # ----------------------------
    # 1. Compute rating-weighted profiles
    # ----------------------------
    user_cat = df_display.pivot_table(
        index="userId",
        columns="category",
        values="rating",
        aggfunc="sum",
        fill_value=0
    )

    rating_counts = df_display.groupby("userId")["rating"].count()
    user_cat["avg_rating_mass"] = user_cat.sum(axis=1) / rating_counts

    # Center ratings per user to avoid compression across users
    user_cat_centered = user_cat[['neutral', 'mildly_political', 'extreme']].sub(
        user_cat[['neutral', 'mildly_political', 'extreme']].mean(axis=1),
        axis=0
    )

    # Extremeness score relative to user's own rating pattern
    user_cat["extremeness"] = (
        user_cat_centered["extreme"] + 0.5 * user_cat_centered["mildly_political"]
    ).div(user_cat_centered.abs().sum(axis=1) + 1e-8)  # avoid div0
    # scale to [0,1]
    user_cat["extremeness"] = (user_cat["extremeness"] - user_cat["extremeness"].min()) / (
        user_cat["extremeness"].max() - user_cat["extremeness"].min() + 1e-8
    )

    # ----------------------------
    # 2. Base scatter
    # ----------------------------
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

    # ----------------------------
    # 3. X-axis segmentation: Neutral / Mild / Extreme
    # ----------------------------
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

    # ----------------------------
    # 4. Highlight selected user
    # ----------------------------
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
