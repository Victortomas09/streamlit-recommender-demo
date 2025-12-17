import streamlit as st
import pandas as pd

def show_pmf_block(user_latent, movie_latent, users_idx, movies_idx):
    st.subheader(f"Method: Probabilistic Matrix Factorization - PMF")

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

        st.latex(
            r"""
            \text{PMF models ratings as Gaussian-distributed:} \\
            """
        )

        st.latex(
            r"""
            p(R | P, Q, \sigma^2) = \prod_{(u,i)\in\mathcal{K}} 
            \mathcal{N}(r_{ui} | p_u^{T} q_i, \sigma^2) \\
            """
        )


        st.latex(
            r"""
            \text{with Gaussian priors on } P \text{ and } Q: \\
            p(P | \sigma_P^2) = \prod_u \mathcal{N}(p_u | 0, \sigma_P^2 I), \quad
            p(Q | \sigma_Q^2) = \prod_i \mathcal{N}(q_i | 0, \sigma_Q^2 I) \\
            """
        )

        st.latex(
            r"""
            \text{Learning is done by maximizing the log-posterior over } P, Q. \\
            """
        )


    col1, col2 = st.columns(2)
    with col1:
        st.write("User factors (P):")
        st.dataframe(pd.DataFrame(user_latent, index=users_idx))
    with col2:
        st.write("Item factors (Q):")
        st.dataframe(pd.DataFrame(movie_latent, index=movies_idx))
