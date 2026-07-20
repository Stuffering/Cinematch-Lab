"""Minimal Streamlit entry point for CineMatch Lab."""

import streamlit as st

from cinematch import __version__

st.set_page_config(page_title="CineMatch Lab", page_icon="🎬", layout="wide")

st.title("CineMatch Lab")
st.caption(f"Version {__version__}")
st.write("Offline movie recommendation and user behavior analysis.")

st.markdown(
    """
    CineMatch Lab includes:

    - Classical recommendation methods: item-CF, matrix factorization,
      content-based, and hybrid recommendation
    - Predictive models: supervised Ridge regression and neural embeddings
    - User behavior analysis: clustering and anomaly detection
    - Engineering workflow: recommendation strategy routing and reusable model artifacts
    """
)

st.info(
    "Run the project scripts from the command line for full model training "
    "and evaluation workflows."
)
