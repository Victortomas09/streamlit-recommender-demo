import streamlit as st
import pandas as pd
import numpy as np

def show_svd_block(user_latent, movie_latent, s, latent_factors, users_idx, movies_idx):
    st.subheader(f"Method: Singular Value Decomposition - SVD")

    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"R = U \Sigma V^{T}")
        st.latex(r"\hat{R} \approx U_k \Sigma_k V_k^{T}")
        st.latex(r"\hat{R} \approx (U_k \sqrt{\Sigma_k})(V_k \sqrt{\Sigma_k})^{T}")

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
        st.dataframe(
            pd.DataFrame(
                np.diag(s[:latent_factors]),
                columns=[f"dim{i+1}" for i in range(latent_factors)]
            )
        )

    with col3:
        st.write("Item factors (V√Σ):")
        st.markdown(f"**Item latent matrix shape:** {movie_latent.shape}")
        st.dataframe(pd.DataFrame(movie_latent, index=movies_idx))

    # Explained Variance
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

    explained_variance_ratio = (s**2) / np.sum(s**2)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    st.markdown(
        f"##### Total Explained Variance (Top {latent_factors} factors): "
        f"{cumulative_explained_variance[latent_factors-1]:.4f}"
    )
