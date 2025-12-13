import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np

# --- IMPORT THE NEW INFERENCE ENGINE ---
from ml.predict import model_service 

st.set_page_config(layout="wide", page_title="Portfolio Overview", page_icon="📈")

st.title("📈 Executive Project Portfolio Dashboard")
st.markdown("A high-level overview of all active projects, using **Live ML Inference** to assess geospatial and financial risk.")

# --- BATCH PREDICTION LOGIC ---
@st.cache_data
def load_and_predict_portfolio():
    """
    Loads raw data and uses the saved XGBoost models to predict risks for ALL projects at once.
    """
    try:
        df = pd.read_csv('projects_data.csv')
    except FileNotFoundError:
        st.error("🚨 'projects_data.csv' not found.")
        return pd.DataFrame()

    # 1. Clean Data
    df.dropna(subset=['lat', 'lon'], inplace=True)
    
    # 2. Preprocess (One-Hot Encoding) to match training
    df_processed = pd.get_dummies(df, columns=['project_type', 'terrain'])
    
    # 3. Align Features (Crucial: Use feature names from the saved model)
    # This prevents "Shape Mismatch" errors
    required_features = model_service.feature_names
    df_aligned = df_processed.reindex(columns=required_features, fill_value=0)
    df_aligned = df_aligned.astype(float)

    # 4. Batch Predict (Using the loaded engine)
    df['predicted_delay'] = model_service.model_timeline.predict(df_aligned)
    df['predicted_cost_overrun'] = model_service.model_cost.predict(df_aligned)

    # 5. Calculate Severity Score (Vectorized for speed)
    # Normalize: Assume 365 days and 500 Lakhs are max risk benchmarks
    d_score = np.minimum(df['predicted_delay'] / 365, 1.0) * 100
    c_score = np.minimum(df['predicted_cost_overrun'] / 500, 1.0) * 100
    df['severity_score'] = (0.6 * d_score) + (0.4 * c_score)

    # 6. Categorize
    def categorize(score):
        if score > 60: return "High Risk"
        elif score > 30: return "Medium Risk"
        return "Low Risk"
    
    df['risk_level'] = df['severity_score'].apply(categorize)
    return df

df = load_and_predict_portfolio()

if df.empty:
    st.stop()

# --- TOP LEVEL METRICS ---
st.markdown("---")
total_projects = len(df)
high_risk_projects = len(df[df['risk_level'] == 'High Risk'])

total_cost_overrun = df['predicted_cost_overrun'].sum()
at_risk_capital = df[df['risk_level'] == 'High Risk']['predicted_cost_overrun'].sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Active Projects", f"{total_projects}")
kpi2.metric("High-Risk Projects", f"{high_risk_projects}", delta=f"{(high_risk_projects / total_projects):.1%} of total", delta_color="inverse")
kpi3.metric("Total Predicted Overrun", f"₹{total_cost_overrun / 100:.2f} Cr")
kpi4.metric("Capital at High Risk", f"₹{at_risk_capital / 100:.2f} Cr", help="Sum of predicted overruns for high-risk projects.")
st.markdown("---")

# --- VISUALIZATIONS ---
col1, col2 = st.columns([2, 1])

# 1. GEOSPATIAL MAP
with col1:
    st.subheader("🌍 Geospatial Risk Map")
    st.caption("Projects colored by Risk Level (Red = High Risk)")
    
    # Center map on average lat/lon
    avg_lat = df['lat'].mean()
    avg_lon = df['lon'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5)

    def get_color(risk_level):
        if risk_level == 'High Risk': return 'red'
        elif risk_level == 'Medium Risk': return 'orange'
        else: return 'green'

    for idx, row in df.iterrows():
        tooltip_text = (f"<b>Project:</b> {row['project_type']}<br>"
                        f"<b>Risk:</b> {row['risk_level']} (Score: {row['severity_score']:.0f})<br>"
                        f"<b>Delay:</b> {row['predicted_delay']:.0f} days<br>"
                        f"<b>Overrun:</b> ₹ {row['predicted_cost_overrun']:.1f} L")
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']], 
            radius=8, 
            color=get_color(row['risk_level']),
            fill=True, 
            fill_color=get_color(row['risk_level']), 
            fill_opacity=0.7, 
            tooltip=tooltip_text
        ).add_to(m)
        
    st_folium(m, use_container_width=True, height=500)

# 2. CHARTS
with col2:
    st.subheader("📊 Risk Analytics")
    
    # Donut Chart
    risk_counts = df['risk_level'].value_counts()
    fig = px.pie(
        values=risk_counts.values, 
        names=risk_counts.index, 
        title="Portfolio Risk Distribution", 
        hole=0.4,
        color=risk_counts.index, 
        color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("🔥 Terrain vs Project Type")
    pivot = pd.pivot_table(df, values='predicted_delay', index='project_type', columns='terrain', aggfunc='mean')
    fig_heatmap = px.imshow(
        pivot, 
        text_auto=".0f", 
        color_continuous_scale='Reds', 
        title="Avg Delay (Days) by Context"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)