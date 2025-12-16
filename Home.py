import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb  # <--- CRITICAL IMPORT ADDED

# --- IMPORT INFERENCE ENGINE ---
from ml.predict import model_service 

st.set_page_config(
    page_title="INFRALYTICS | Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 CUSTOM CSS FOR "PRO" UI ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Card Styling */
    .dashboard-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    
    /* KPI Value Styling */
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #1f2937;
    }
    .kpi-label {
        font-size: 14px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk Badges */
    .risk-high { color: #dc2626; background-color: #fef2f2; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }
    .risk-med { color: #d97706; background-color: #fffbeb; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }
    
    /* Navigation Cards */
    .nav-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        transition: transform 0.2s;
        cursor: pointer;
    }
    .nav-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        border-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIC: BATCH PREDICTION ---
def calculate_severity_batch(delay, cost):
    d_score = np.minimum(delay / 365, 1.0) * 100
    c_score = np.minimum(cost / 500, 1.0) * 100
    return (0.6 * d_score) + (0.4 * c_score)

@st.cache_data
def load_dashboard_data():
    try:
        df = pd.read_csv('projects_data.csv')
    except:
        return pd.DataFrame() # Return empty if fails

    # Preprocessing
    df_proc = pd.get_dummies(df, columns=['project_type', 'terrain'])
    features = model_service.feature_names
    
    # Align
    df_aligned = df_proc.reindex(columns=features, fill_value=0).astype(float)
    
    # --- CRITICAL FIX START ---
    # Convert DataFrame to DMatrix (Native XGBoost Format)
    dmatrix_data = xgb.DMatrix(df_aligned)
    
    # Batch Predict using the DMatrix
    df['pred_delay'] = model_service.model_timeline.predict(dmatrix_data)
    df['pred_cost'] = model_service.model_cost.predict(dmatrix_data)
    # --- CRITICAL FIX END ---
    
    # Score
    df['severity'] = calculate_severity_batch(df['pred_delay'], df['pred_cost'])
    df['risk_level'] = df['severity'].apply(lambda x: "High" if x > 60 else ("Medium" if x > 30 else "Low"))
    
    return df

df = load_dashboard_data()

# --- HERO HEADER ---
col_logo, col_title, col_user = st.columns([1, 6, 2])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3061/3061341.png", width=80) # Placeholder Icon
with col_title:
    st.title("INFRALYTICS")
    st.markdown("**AI-Powered Project Risk Intelligence Platform**")
with col_user:
    st.caption("Logged in as: **Admin**")
    st.caption("Last Update: **Live** 🟢")

st.markdown("---")

if df.empty:
    st.error("⚠️ No data found. Please add 'projects_data.csv' to the root folder.")
    st.stop()

# --- 📊 ROW 1: EXECUTIVE KPI CARDS ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total = len(df)
high_risk = len(df[df['risk_level'] == 'High'])
capital_risk = df[df['risk_level'] == 'High']['pred_cost'].sum()
avg_delay = df['pred_delay'].mean()

# Helper for KPI Card
def kpi_card(col, title, value, subtext, color="black"):
    col.markdown(f"""
    <div class="dashboard-card" style="border-left: 5px solid {color};">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        <div style="font-size:12px; color:gray;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(kpi1, "Total Projects", f"{total}", "Active Portfolio", "#3b82f6")
kpi_card(kpi2, "High Risk", f"{high_risk}", f"{(high_risk/total)*100:.1f}% of Portfolio", "#ef4444")
kpi_card(kpi3, "Capital at Risk", f"₹{capital_risk/100:.1f} Cr", "Projected Overruns", "#f59e0b")
kpi_card(kpi4, "Avg Delay", f"{int(avg_delay)} Days", "Across all regions", "#10b981")

# --- 🚀 ROW 2: DASHBOARD SNAPSHOTS (Charts directly on home) ---
col_chart1, col_chart2, col_actions = st.columns([2, 2, 1.5])

with col_chart1:
    st.markdown("##### 🌍 Portfolio Health Distribution")
    # Sunburst Chart: Type -> Risk Level
    fig_sun = px.sunburst(
        df, 
        path=['project_type', 'risk_level'], 
        values='pred_cost',
        color='risk_level',
        color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'},
        height=300
    )
    fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    st.plotly_chart(fig_sun, use_container_width=True)

with col_chart2:
    st.markdown("##### 📉 Cost vs. Delay Correlation")
    # Scatter Plot
    fig_scat = px.scatter(
        df, 
        x='pred_delay', 
        y='pred_cost', 
        color='risk_level',
        size='severity',
        hover_data=['project_type'],
        color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'},
        height=300
    )
    fig_scat.update_layout(margin=dict(t=0, l=0, r=0, b=0), xaxis_title="Delay (Days)", yaxis_title="Cost (Lakhs)")
    st.plotly_chart(fig_scat, use_container_width=True)

with col_actions:
    st.markdown("##### ⚡ Quick Actions")
    
    # Custom HTML Buttons that look like cards
    st.markdown("""
    <div class="nav-card">
        <div style="font-weight:bold; font-size:16px;">🔮 Simulator</div>
        <p style="font-size:12px; margin:0;">Test 'What-If' scenarios for new projects.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Simulator >", key="btn_sim", use_container_width=True):
        st.switch_page("pages/1_🔮_Project_Predictor.py")
    
    st.write("") # Spacer

    st.markdown("""
    <div class="nav-card">
        <div style="font-weight:bold; font-size:16px;">👥 Vendor Intel</div>
        <p style="font-size:12px; margin:0;">Analyze contractor performance history.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Analyze Vendors >", key="btn_vend", use_container_width=True):
        st.switch_page("pages/3_👥_Vendor_Analysis.py")

# --- 🚨 ROW 3: LIVE ALERTS FEED ---
st.markdown("### 🚨 Recent Critical Alerts")
critical_projects = df[df['risk_level'] == 'High'].sort_values('severity', ascending=False).head(3)

if critical_projects.empty:
    st.success("✅ System Status Normal: No Critical Risks Detected.")
else:
    for idx, row in critical_projects.iterrows():
        with st.expander(f"🔴 ALERT: {row['project_type']} in {row['terrain']} Terrain (Severity: {int(row['severity'])})", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Delay", f"{int(row['pred_delay'])} Days")
            c2.metric("Cost Overrun", f"₹ {int(row['pred_cost'])} Lakhs")
            c3.write(f"**AI Recommendation:** High historical delay probability in {row['terrain']} terrain. Immediate audit advised.")