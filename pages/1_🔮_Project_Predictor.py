# pages/01_🔮_Project_Predictor.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from upgraded_model_logic import train_models_and_explainer, get_predictions_and_hotspots
import lime
import lime.lime_tabular


# --- Caching Models and Data ---
@st.cache_resource
def load_models():
    """Loads and caches the trained models, explainer, and feature names."""
    model_timeline, model_cost, explainer, feature_names = train_models_and_explainer()
    return model_timeline, model_cost, explainer, feature_names


@st.cache_data
def load_historical_data():
    """Loads and caches the historical project data."""
    return pd.read_csv('projects_data.csv')


# Load all necessary assets
model_timeline, model_cost, explainer_timeline, feature_names = load_models()
historical_df = load_historical_data()

# --- UI Sidebar ---
st.sidebar.header("Enter Project Details")
project_type = st.sidebar.selectbox('Project Type', historical_df['project_type'].unique())
terrain = st.sidebar.selectbox('Terrain Type', historical_df['terrain'].unique())
material_cost = st.sidebar.number_input('Material Cost (in Crores)', min_value=50, max_value=1000, value=200)
vendor_rating = st.sidebar.slider('Vendor Performance Rating (1=Worst, 5=Best)', 1, 5, 3)
historical_delays = st.sidebar.number_input('Number of Historical Delays (Similar Projects)', min_value=0, max_value=50,
                                            value=8)

# --- Main Page ---
st.title("🔮 Project Predictor & Simulator")
st.write("Input project details in the sidebar to get a real-time risk assessment.")

if st.sidebar.button('Run Prediction', type="primary"):
    # 1. Get Prediction
    input_data = {'project_type': project_type, 'terrain': terrain, 'material_cost_crore': material_cost,
                  'vendor_performance_rating': vendor_rating, 'historical_delays_project_count': historical_delays}
    results = get_predictions_and_hotspots(input_data, model_timeline, model_cost, explainer_timeline, feature_names)
    predicted_delay = results['predicted_delay']
    predicted_cost_overrun = results['predicted_cost_overrun']
    severity_score = results['severity_score']

    # --- Display KPIs ---
    st.subheader("Prediction Summary")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Predicted Timeline Delay", f"{predicted_delay:.0f} Days")
    kpi2.metric("Predicted Cost Overrun", f"₹{predicted_cost_overrun:.2f} Lakhs")
    kpi3.metric("Overall Risk Severity", f"{severity_score:.0f} / 100")
    st.markdown("---")

    # --- Visualizations ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Risk Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=severity_score, title={'text': "Severity Score"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "rgba(0,0,0,0.3)"},
                   'steps': [{'range': [0, 40], 'color': "#3D9970"}, {'range': [40, 70], 'color': "#FFDC00"},
                             {'range': [70, 100], 'color': "#FF4136"}]}))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Prediction vs. Historical Average")
        similar_projects = historical_df[
            (historical_df['project_type'] == project_type) & (historical_df['terrain'] == terrain)]
        avg_delay = similar_projects['actual_timeline_delay_days'].mean()
        chart_data = pd.DataFrame({'Category': ["Your Project's Prediction", "Historical Average"],
                                   'Delay (Days)': [predicted_delay, avg_delay]})
        fig_bar = px.bar(chart_data, y='Category', x='Delay (Days)', text_auto=True, orientation='h',
                         title="Predicted Delay Comparison")
        fig_bar.update_layout(yaxis_title=None, height=350)
        fig_bar.update_traces(marker_color=['#FF4136', 'lightgrey'], texttemplate='%{x:.0f} days',
                              textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # --- LIME EXPLANATION SECTION (REPLACES THE OLD SHAP SECTION) ---
    st.subheader("LIME Explanation: Key Prediction Factors")
    st.write(
        "This chart shows the features that had the most impact on this specific prediction, as determined by the LIME algorithm.")

    # Create a LIME explainer
    X_train = pd.get_dummies(
        historical_df.drop(['actual_timeline_delay_days', 'cost_overrun_lakhs', 'lat', 'lon'], axis=1, errors='ignore'))
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values, feature_names=feature_names,
        class_names=['timeline_delay'], mode='regression'
    )

    # Explain the single prediction
    instance_to_explain = results['input_aligned'].iloc[0].values
    explanation = explainer.explain_instance(
        instance_to_explain, model_timeline.predict, num_features=10
    )

    # Convert LIME explanation to a DataFrame for plotting
    lime_results = pd.DataFrame(explanation.as_list(), columns=['feature', 'weight'])
    lime_results['positive'] = lime_results['weight'] > 0

    # Create a Plotly bar chart
    fig_lime = px.bar(
        lime_results.sort_values(by='weight'), x='weight', y='feature', orientation='h',
        color='positive', color_discrete_map={True: '#FF4136', False: '#3D9970'},
        title="Factors Pushing Prediction Higher (Red) or Lower (Green)"
    )
    fig_lime.update_layout(showlegend=False, yaxis_title=None, xaxis_title="Impact on Prediction (Weight)")
    st.plotly_chart(fig_lime, use_container_width=True)

else:
    st.info("Please enter project details in the sidebar and click 'Run Prediction' to see the analysis.")