import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np
import xgboost as xgb

# --- IMPORT THE INFERENCE ENGINE ---
from ml.predict import model_service 

st.set_page_config(layout="wide", page_title="Portfolio Overview", page_icon="📈")

# --- SIDEBAR: DATA CONTROLS ---
st.sidebar.title("📂 Data Management")

# 1. Template Downloader (The "Usability Flex")
# We create a sample CSV in memory so the user knows what to upload
@st.cache_data
def get_template_csv():
    df_template = pd.DataFrame({
        "project_id": ["P001", "P002"],
        "project_type": ["Overhead Line", "Substation"],
        "terrain": ["Hilly", "Plain"],
        "lat": [28.61, 19.07],
        "lon": [77.20, 72.87],
        "vendor_performance_rating": [3, 5]
    })
    return df_template.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="📥 Download CSV Template",
    data=get_template_csv(),
    file_name="infralytics_template.csv",
    mime="text/csv",
    help="Use this format to upload your own project data."
)

st.sidebar.markdown("---")

# 2. File Uploader
uploaded_file = st.sidebar.file_uploader("Upload New Portfolio", type=["csv"])

# --- MAIN APP LOGIC ---
st.title("📈 Executive Project Portfolio Dashboard")
st.markdown("A high-level overview of active projects using **Live ML Inference**.")

# Indicator of what data is running
if uploaded_file:
    st.info("✅ Running Analysis on **Uploaded User Data**")
else:
    st.warning("⚠️ Running Analysis on **Demo Data** (Upload a CSV to analyze real projects)")

# --- BATCH PREDICTION LOGIC ---
# We removed @st.cache_data here because uploaded files change often
def load_and_predict_portfolio(file_obj):
    """
    Loads data (file or default), aligns features, and predicts risks.
    """
    try:
        if file_obj is not None:
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_csv('projects_data.csv')
    except Exception as e:
        st.error(f"🚨 Error reading file: {e}")
        return pd.DataFrame()

    # --- VALIDATION (The "Engineer" check) ---
    required_cols = ['project_type', 'terrain', 'lat', 'lon']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Uploaded file is missing columns: {missing}")
        st.stop()
        
    # 1. Clean Data
    df.dropna(subset=['lat', 'lon'], inplace=True)
    
    # 2. Preprocess (One-Hot Encoding)
    # We use 'try' because user data might have new categories unseen by the model
    try:
        df_processed = pd.get_dummies(df, columns=['project_type', 'terrain'])
    except Exception as e:
        st.error(f"Data Processing Error: {e}")
        st.stop()
    
    # 3. Align Features (Crucial for XGBoost)
    required_features = model_service.feature_names
    df_aligned = df_processed.reindex(columns=required_features, fill_value=0)
    df_aligned = df_aligned.astype(float)

    # 4. Batch Predict (Native Booster Mode)
    try:
        dmatrix_data = xgb.DMatrix(df_aligned)
        df['predicted_delay'] = model_service.model_timeline.predict(dmatrix_data)
        df['predicted_cost_overrun'] = model_service.model_cost.predict(dmatrix_data)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    # 5. Calculate Severity Score
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

# Run the function with the uploaded file (or None)
df = load_and_predict_portfolio(uploaded_file)

if df.empty:
    st.stop()

# --- DASHBOARD METRICS ---
st.markdown("---")
total_projects = len(df)
high_risk_projects = len(df[df['risk_level'] == 'High Risk'])
total_cost_overrun = df['predicted_cost_overrun'].sum()
at_risk_capital = df[df['risk_level'] == 'High Risk']['predicted_cost_overrun'].sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Active Projects", f"{total_projects}")
kpi2.metric("High-Risk Projects", f"{high_risk_projects}", delta=f"{(high_risk_projects / total_projects):.1%} of total", delta_color="inverse")
kpi3.metric("Total Predicted Overrun", f"₹{total_cost_overrun / 100:.2f} Cr")
kpi4.metric("Capital at High Risk", f"₹{at_risk_capital / 100:.2f} Cr")
st.markdown("---")

# --- VISUALIZATIONS ---
col1, col2 = st.columns([2, 1])

# 1. GEOSPATIAL MAP
with col1:
    st.subheader("🌍 Geospatial Risk Map")
    
    avg_lat = df['lat'].mean()
    avg_lon = df['lon'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5)

    def get_color(risk_level):
        return {'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'}.get(risk_level, 'gray')

    for idx, row in df.iterrows():
        tooltip_text = (f"<b>Type:</b> {row['project_type']}<br>"
                        f"<b>Risk:</b> {row['risk_level']}<br>"
                        f"<b>Delay:</b> {row['predicted_delay']:.0f} days")
        
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
    st.subheader("🔥 Contextual Risk")
    pivot = pd.pivot_table(df, values='predicted_delay', index='project_type', columns='terrain', aggfunc='mean')
    fig_heatmap = px.imshow(
        pivot, text_auto=".0f", color_continuous_scale='Reds', aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)