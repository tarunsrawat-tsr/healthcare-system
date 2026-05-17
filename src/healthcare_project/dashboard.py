import streamlit as st
import requests
import pandas as pd

# API Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Claimsight Dashboard", layout="wide")
st.title("Claimsight: Anomaly Detection Dashboard")

# Fetch and Display Summary Statistics
st.header("Pipeline Statistics")
try:
    stats_response = requests.get(f"{API_URL}/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Claims Scored", f"{stats['total_scored']:,}")
        col2.metric("Flagged by Peers (Z-Score)", f"{stats['flagged_by_peer']:,}")
        col3.metric("Flagged by Isolation Forest", f"{stats['flagged_by_iforest']:,}")
        col4.metric("Flagged by Both", f"{stats['flagged_by_both']:,}")
    else:
        st.error("Failed to load statistics from the API.")
except requests.exceptions.ConnectionError:
    st.error("Could not connect to the API. Ensure FastAPI is running on port 8000.")

st.divider()

# Controls & Filters
st.header("Top Anomalies")
col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    limit = st.slider("Number of results to display", min_value=10, max_value=500, value=50)
with col_filter2:
    require_both = st.checkbox("Require both Z-Score and Isolation Forest flags", value=False)

# Fetch and Display Anomaly Data
try:
    params = {"limit": limit, "require_both_flags": require_both}
    anomalies_response = requests.get(f"{API_URL}/anomalies", params=params)
    
    if anomalies_response.status_code == 200:
        data = anomalies_response.json()
        if data:
            df = pd.DataFrame(data)
            
            # Formatting the dataframe for better UI presentation
            display_cols = ['provider_id', 'drg', 'state', 'charges', 'payments', 'markup_ratio', 'robust_z', 'iforest_flag']
            df_display = df[display_cols].copy()
            
            # Format currency and numbers
            df_display['charges'] = df_display['charges'].apply(lambda x: f"${x:,.2f}")
            df_display['payments'] = df_display['payments'].apply(lambda x: f"${x:,.2f}")
            df_display['markup_ratio'] = df_display['markup_ratio'].apply(lambda x: f"{x:.2f}x")
            df_display['robust_z'] = df_display['robust_z'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No anomalies found matching the current criteria.")
except requests.exceptions.ConnectionError:
    st.warning("Data table unavailable while API is offline.")