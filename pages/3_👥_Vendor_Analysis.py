import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- IMPORT THE NEW INFERENCE ENGINE ---
from ml.predict import model_service 

st.set_page_config(page_title="Vendor Analysis", page_icon="👥", layout="wide")

st.title("👥 Vendor Performance Intelligence")
st.markdown("Analyze supply chain risks by evaluating vendor performance against AI-predicted delays.")

# --- BATCH PREDICTION LOGIC ---
@st.cache_data
def load_vendor_data():
    try:
        df = pd.read_csv('projects_data.csv')
    except FileNotFoundError:
        st.error("🚨 'projects_data.csv' not found.")
        return pd.DataFrame()

    # 1. Preprocess
    df_processed = pd.get_dummies(df, columns=['project_type', 'terrain'])
    
    # 2. Align Features
    required_features = model_service.feature_names
    df_aligned = df_processed.reindex(columns=required_features, fill_value=0).astype(float)

    # 3. Batch Predict
    df['predicted_delay'] = model_service.model_timeline.predict(df_aligned)
    df['predicted_cost_overrun'] = model_service.model_cost.predict(df_aligned)

    # 4. Score Severity
    d_score = np.minimum(df['predicted_delay'] / 365, 1.0) * 100
    c_score = np.minimum(df['predicted_cost_overrun'] / 500, 1.0) * 100
    df['severity_score'] = (0.6 * d_score) + (0.4 * c_score)
    
    return df

df = load_vendor_data()

if df.empty:
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Vendors")
available_ratings = sorted(df['vendor_performance_rating'].unique())
selected_rating = st.sidebar.selectbox("Select Vendor Rating (1-5)", available_ratings, index=2)

# Filter Data
vendor_df = df[df['vendor_performance_rating'] == selected_rating]

# --- MAIN DASHBOARD ---
st.subheader(f"📊 Analysis for Vendors with Rating: {selected_rating}/5")

# Metrics
avg_delay = vendor_df['predicted_delay'].mean()
avg_cost = vendor_df['predicted_cost_overrun'].mean()
avg_sev = vendor_df['severity_score'].mean()
count = len(vendor_df)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Projects Managed", f"{count}")
k2.metric("Avg. Predicted Delay", f"{int(avg_delay)} Days", delta=f"{int(avg_delay - df['predicted_delay'].mean())} vs Global Avg", delta_color="inverse")
k3.metric("Avg. Cost Overrun", f"₹ {avg_cost:.2f} L")
k4.metric("Risk Severity", f"{int(avg_sev)}/100")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risk Distribution")
    # Donut Chart for this vendor tier
    fig_pie = px.pie(
        vendor_df, 
        names='project_type', 
        values='severity_score', 
        title="Risk Contribution by Project Type",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Performance Matrix")
    # Scatter plot: Delay vs Cost
    fig_scatter = px.scatter(
        vendor_df, 
        x='predicted_delay', 
        y='predicted_cost_overrun', 
        color='project_type', 
        size='severity_score',
        hover_data=['terrain'],
        title="Project Outcomes (Size = Risk Severity)",
        labels={'predicted_delay': 'Predicted Delay (Days)', 'predicted_cost_overrun': 'Cost Overrun (Lakhs)'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Detailed Table
st.subheader(f"📋 Project Details (Rating {selected_rating})")
display_cols = ['project_type', 'terrain', 'predicted_delay', 'predicted_cost_overrun', 'severity_score']
st.dataframe(
    vendor_df[display_cols].sort_values('severity_score', ascending=False).style.format({
        'predicted_delay': '{:.0f}',
        'predicted_cost_overrun': '{:.2f}',
        'severity_score': '{:.0f}'
    }),
    use_container_width=True
)