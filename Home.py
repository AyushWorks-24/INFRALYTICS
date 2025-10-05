import streamlit as st
from upgraded_model_logic import train_models_and_explainer, calculate_severity_score
import pandas as pd

st.set_page_config(
    page_title="POWERGRID AI Command Center",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_models():
    model_timeline, model_cost, _, feature_names = train_models_and_explainer()
    return model_timeline, model_cost, feature_names

model_timeline, model_cost, feature_names = load_models()

@st.cache_data
def load_and_predict_portfolio_data():
    df = pd.read_csv('projects_data.csv')
    df_processed = pd.get_dummies(df, columns=['project_type', 'terrain'])
    df_aligned = df_processed.reindex(columns=feature_names, fill_value=0)
    df['predicted_delay'] = model_timeline.predict(df_aligned)
    df['predicted_cost_overrun'] = model_cost.predict(df_aligned)
    df['severity_score'] = df.apply(
        lambda row: calculate_severity_score(row['predicted_delay'], row['predicted_cost_overrun']),
        axis=1
    )
    def categorize_risk(score):
        if score > 60: return "High Risk"
        elif score > 30: return "Medium Risk"
        else: return "Low Risk"
    df['risk_level'] = df['severity_score'].apply(categorize_risk)
    return df

df = load_and_predict_portfolio_data()

st.title("⚡ AI-Powered Project Command Center")
st.markdown("Welcome to the central hub for predicting and managing project risks for POWERGRID. Use the navigation on the left to access specialized dashboards.")
st.markdown("---")

total_projects = len(df)
high_risk_projects = len(df[df['risk_level'] == 'High Risk'])
avg_severity = df['severity_score'].mean()
at_risk_capital = df[df['risk_level'] == 'High Risk']['predicted_cost_overrun'].sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Active Projects", f"{total_projects}")
kpi2.metric("High-Risk Projects", f"{high_risk_projects}")
kpi3.metric("Avg. Portfolio Severity", f"{avg_severity:.0f} / 100")
kpi4.metric("Capital at High Risk", f"₹{at_risk_capital / 100:.2f} Cr")

st.markdown("---")
st.subheader("Explore Your Dashboards")
col1, col2 = st.columns(2)

with col1:
    st.info("#### 🔮 Project Predictor & Simulator")
    st.write("Analyze a single new project. Input its details to get real-time predictions on delays, costs, and risk factors. Run 'what-if' scenarios to see how changes can mitigate risk.")
    st.page_link("pages/1_🔮_Project_Predictor.py", label="Go to Predictor", icon="🔮")
    st.info("#### 👥 Vendor Performance Analysis")
    st.write("Deep-dive into vendor performance. Identify which vendors are associated with higher risks and analyze their project history to make informed partnership decisions.")
    st.page_link("pages/3_👥_Vendor_Analysis.py", label="Analyze Vendors", icon="👥")

with col2:
    st.info("#### 📈 Portfolio Overview")
    st.write("Get a 30,000-foot view of the entire project portfolio. Use the geospatial map and interactive charts to identify risk concentrations and monitor overall health.")
    st.page_link("pages/2_📈_Portfolio_Overview.py", label="View Portfolio", icon="📈")
    st.info("#### 🗺️ Regional Risk Analysis")
    st.write("Analyze how geographical factors like terrain impact project outcomes. Compare risk levels and average delays across different operational environments.")
    st.page_link("pages/4_🗺️_Regional_Analysis.py", label="Analyze Regions", icon="🗺️")