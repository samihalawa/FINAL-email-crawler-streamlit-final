import streamlit as st
from health_checks import display_health_dashboard

# Add unique key to avoid duplicate widget IDs
st.set_page_config(page_title="Health Dashboard", page_icon="🏥")

# Display the health dashboard
display_health_dashboard()