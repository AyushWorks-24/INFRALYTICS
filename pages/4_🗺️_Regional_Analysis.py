import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- IMPORT THE NEW INFERENCE ENGINE ---
from ml.predict import model_service 

st.set_page_config(page_title="Regional Analysis", page_icon="🗺️", layout="wide")

st.title("🗺️ Regional & Terrain Analysis")
st.markdown("Investigate how geographical factors and terrain types impact project timelines and costs.")

# --- BATCH PREDICTION LOGIC ---
@st.cache_data
def load_regional_data():
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
    
    # 4. Severity
    d_score = np.minimum(df['predicted_delay'] / 365, 1.0) * 100
    c_score = np.minimum(df['predicted_cost_overrun'] / 500, 1.0) * 100
    df['severity_score'] = (0.6 * d_score) + (0.4 * c_score)
    
    return df

df = load_regional_data()

if df.empty:
    st.stop()

# --- TERRAIN COMPARISON ---
st.subheader("⛰️ Impact of Terrain on Delays")

# Group by Terrain
terrain_stats = df.groupby('terrain')[['predicted_delay', 'predicted_cost_overrun', 'severity_score']].mean().reset_index()
terrain_stats = terrain_stats.sort_values('predicted_delay', ascending=False)

col1, col2 = st.columns(2)

with col1:
    # Bar Chart: Delay by Terrain
    fig_bar = px.bar(
        terrain_stats,
        x='predicted_delay',
        y='terrain',
        orientation='h',
        color='severity_score',
        color_continuous_scale='Reds',
        title="Average Predicted Delay by Terrain",
        text_auto='.0f'
    )
    fig_bar.update_layout(xaxis_title="Avg Delay (Days)", yaxis_title="Terrain")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    # Box Plot: Variance Analysis
    fig_box = px.box(
        df, 
        x='terrain', 
        y='predicted_delay',
        color='terrain',
        title="Delay Variability Distribution",
        points="all" # Show individual points
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- REGIONAL INSIGHTS ---
st.markdown("---")
st.subheader("🔍 Comparative Analysis")

# Pivot Table Heatmap
st.write("Cross-Analysis: Average Cost Overrun (₹ Lakhs)")
pivot_cost = pd.pivot_table(
    df, 
    values='predicted_cost_overrun', 
    index='project_type', 
    columns='terrain', 
    aggfunc='mean'
)

fig_heat = px.imshow(
    pivot_cost, 
    text_auto=".1f", 
    color_continuous_scale='Viridis', 
    aspect="auto",
    title="Heatmap: Cost Overrun by Type & Terrain"
)
st.plotly_chart(fig_heat, use_container_width=True)

# --- AI INSIGHT ---
st.info(f"💡 **AI Insight:** Projects in **{terrain_stats.iloc[0]['terrain']}** terrain show the highest average risk. "
        f"Consider increasing contingency budgets for this region by **{int(terrain_stats.iloc[0]['severity_score'])}%**.")