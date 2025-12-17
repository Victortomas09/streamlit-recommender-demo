import streamlit as st
import pandas as pd

def show_als_block(user_latent, movie_latent, users_idx, movies_idx):
    st.subheader(f"Method: Alternating Least Squares - ALS")

    col1, col2 = st.columns(2)

    with col1:

        st.latex(r"\hat{R} \approx P Q^{T}")

        st.latex(
            r"""
            \begin{array}{l}
            \scriptsize
            \text{Where:} \\
            \scriptsize
            \hat{R} \text{: reconstructed (predicted) ratings} \\
            \scriptsize
            P \text{: user latent feature matrix (users × latent factors)} \\
            \scriptsize
            Q \text{: item latent feature matrix (items × latent factors)} \\
            \scriptsize
            \end{array}
            """
        )
    
    with col2:
        st.latex(r"\text{ALS learns } P \text{ and } Q \text{ by alternatingly minimizing:}")

        st.latex(
            r"""
            \min_{P, Q} \sum_{(u,i)\in \mathcal{K}} (r_{ui} - p_u^{T} q_i)^2  + \lambda ( \|p_u\|^2 + \|q_i\|^2 ) \\
            """
        )

        st.latex(
            r"""
            \text{Fix } Q \text{ and solve for } P, \text{ then fix } P \text{ and solve for } Q. \\
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        st.write("User factors (P):")
        st.dataframe(pd.DataFrame(user_latent, index=users_idx))
    with col2:
        st.write("Item factors (Q):")
        st.dataframe(pd.DataFrame(movie_latent, index=movies_idx))
