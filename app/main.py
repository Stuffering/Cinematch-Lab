"""Minimal Streamlit entry point for CineMatch Lab."""

import streamlit as st

from cinematch import __version__

st.set_page_config(page_title="CineMatch Lab", page_icon="🎬", layout="wide")

st.title("CineMatch Lab")
st.caption(f"Version {__version__}")
st.write("Offline movie recommendation and user behavior analysis")

st.info("Stage 1 is ready. Model features will be added in later stages.")

