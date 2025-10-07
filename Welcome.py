import streamlit as st

st.set_page_config(page_title="Welcome to Stocklyze", page_icon="🤖", layout="centered")

# Chat-style intro
st.chat_message("assistant").markdown("👋 Welcome to **Stocklyze**! What are you looking for today?")

col1, col2 = st.columns(2, gap="large")

with col1:
    if st.button("📊 General Analysis (Top 5 Candidates)", use_container_width=True):
        st.switch_page("pages/Stocklyze insights.py")   # update filename if different

with col2:
    if st.button("📈 Individual Stock + News", use_container_width=True):
        st.switch_page("pages/Stock_profile_+_News.py")