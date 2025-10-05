# pages/02_📈_Portfolio_Overview.py

import streamlit as st
import pandas as pd
import plotly.express as px
from upgraded_model_logic import train_models_and_explainer, calculate_severity_score
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
st.title("📈 Executive Project Portfolio Dashboard")
st.markdown("A high-level overview of all active projects, using live ML predictions to assess risk.")

# (All the caching and data loading code remains the same as before...)
@st.cache_resource
def load_models():
    model_timeline, model_cost, _, feature_names = train_models_and_explainer()
    return model_timeline, model_cost, feature_names

model_timeline, model_cost, feature_names = load_models()

@st.cache_data
def load_and_predict_portfolio_data():
    df = pd.read_csv('projects_data.csv')
    df.dropna(subset=['lat', 'lon'], inplace=True)
    df_processed = pd.get_dummies(df, columns=['project_type', 'terrain'])
    df_aligned = df_processed.reindex(columns=feature_names, fill_value=0)
    df['predicted_delay'] = model_timeline.predict(df_aligned)
    df['predicted_cost_overrun'] = model_cost.predict(df_aligned)
    df['severity_score'] = df.apply(lambda row: calculate_severity_score(row['predicted_delay'], row['predicted_cost_overrun']), axis=1)
    def categorize_risk(score):
        if score > 60: return "High Risk"
        elif score > 30: return "Medium Risk"
        else: return "Low Risk"
    df['risk_level'] = df['severity_score'].apply(categorize_risk)
    return df

df = load_and_predict_portfolio_data()

# --- KPIs ---
st.markdown("---")
total_projects = len(df)
high_risk_projects = len(df[df['risk_level'] == 'High Risk'])

# --- THIS IS THE CORRECTED SECTION ---
total_cost_overrun = df['predicted_cost_overrun'].sum()
at_risk_capital = df[df['risk_level'] == 'High Risk']['predicted_cost_overrun'].sum()
# --- END OF CORRECTION ---

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Active Projects", f"{total_projects}")
kpi2.metric("High-Risk Projects", f"{high_risk_projects}", delta=f"{(high_risk_projects / total_projects):.1%} of total", delta_color="inverse")
kpi3.metric("Total Predicted Overrun", f"₹{total_cost_overrun / 100:.2f} Cr")
kpi4.metric("Capital at High Risk", f"₹{at_risk_capital / 100:.2f} Cr", help="Sum of predicted overruns for high-risk projects.")
st.markdown("---")

# --- Main Visualizations ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Geospatial Risk Overview (using OpenStreetMap)")
    m = folium.Map(location=[22.5, 82], zoom_start=5)
    def get_color(risk_level):
        if risk_level == 'High Risk': return 'red'
        elif risk_level == 'Medium Risk': return 'orange'
        else: return 'green'

    for idx, row in df.iterrows():
        tooltip_text = (f"<b>Project:</b> {row['project_type']}<br>"
                        f"<b>Risk:</b> {row['risk_level']} (Score: {row['severity_score']:.0f})<br>"
                        f"<b>Delay:</b> {row['predicted_delay']:.0f} days")
        folium.CircleMarker(
            location=[row['lat'], row['lon']], radius=8, color=get_color(row['risk_level']),
            fill=True, fill_color=get_color(row['risk_level']), fill_opacity=0.7, tooltip=tooltip_text
        ).add_to(m)
    st_folium(m, use_container_width=True)

with col2:
    st.subheader("Risk Distribution")
    risk_counts = df['risk_level'].value_counts()
    fig = px.pie(values=risk_counts.values, names=risk_counts.index, title="Projects by Risk Category", color=risk_counts.index, color_discrete_map={"High Risk": "#FF4136", "Medium Risk": "#FFDC00", "Low Risk": "#3D9970"})
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Risk Heatmap")
    st.write("Average predicted delay by Project Type and Terrain.")
    pivot = pd.pivot_table(df, values='predicted_delay', index='project_type', columns='terrain')
    fig_heatmap = px.imshow(pivot, text_auto=True, color_continuous_scale='Reds', title="Average Delay (Days) Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)