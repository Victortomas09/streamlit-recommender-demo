import streamlit as st

def show_error_metrics(mse, rmse):
    """Display MSE and RMSE with LaTeX equations in three columns."""
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
        st.markdown(f"##### Root Mean Squared Error (RMSE): {rmse:.4f}")
