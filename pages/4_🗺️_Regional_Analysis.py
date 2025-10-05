import streamlit as st
import pandas as pd
import plotly.express as px
from upgraded_model_logic import train_models_and_explainer, calculate_severity_score

st.title("🗺️ Regional & Terrain-Based Risk Analysis")
st.write("Analyze how project outcomes are influenced by their geographical environment.")

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
    df['severity_score'] = df.apply(lambda row: calculate_severity_score(row['predicted_delay'], row['predicted_cost_overrun']), axis=1)
    def categorize_risk(score):
        if score > 60: return "High Risk"
        elif score > 30: return "Medium Risk"
        else: return "Low Risk"
    df['risk_level'] = df['severity_score'].apply(categorize_risk)
    return df

df = load_and_predict_portfolio_data()

st.subheader("Average Severity by Terrain")
st.write("Compare the overall risk associated with different types of terrain.")
avg_severity_by_terrain = df.groupby('terrain')['severity_score'].mean().sort_values(ascending=False)
fig_bar = px.bar(avg_severity_by_terrain, x=avg_severity_by_terrain.index, y=avg_severity_by_terrain.values, labels={'x': 'Terrain Type', 'y': 'Average Severity Score'}, color=avg_severity_by_terrain.index, text_auto='.2s')
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.subheader("Risk Breakdown by Terrain and Project Type")
col1, col2 = st.columns(2)

with col1:
    st.write("Count of High/Medium/Low Risk Projects by Terrain")
    risk_counts = df.groupby(['terrain', 'risk_level']).size().reset_index(name='count')
    fig_hist = px.histogram(risk_counts, x='terrain', y='count', color='risk_level', barmode='group', category_orders={"risk_level": ["Low Risk", "Medium Risk", "High Risk"]}, color_discrete_map={"High Risk": "#FF4136", "Medium Risk": "#FFDC00", "Low Risk": "#3D9970"})
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.write("Average Delay (Days) by Terrain and Project Type")
    pivot = pd.pivot_table(df, values='predicted_delay', index='project_type', columns='terrain')
    fig_heatmap = px.imshow(pivot, text_auto=True, color_continuous_scale='Reds', title="Average Delay Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)