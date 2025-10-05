import streamlit as st
import pandas as pd
import plotly.express as px
from upgraded_model_logic import train_models_and_explainer, calculate_severity_score

st.title("👥 Vendor Performance Analysis")
st.write("Analyze vendor track records to identify top performers and potential risks.")

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
    return df

df = load_and_predict_portfolio_data()

st.sidebar.header("Filters")
vendors = sorted(df['vendor_performance_rating'].unique())
selected_vendor_rating = st.sidebar.selectbox("Select Vendor Rating to Analyze", vendors)

vendor_df = df[df['vendor_performance_rating'] == selected_vendor_rating]

st.subheader(f"Performance for Vendors with Rating: {selected_vendor_rating}/5")
avg_delay, avg_cost_overrun, num_projects, avg_severity = vendor_df['predicted_delay'].mean(), vendor_df['predicted_cost_overrun'].mean(), len(vendor_df), vendor_df['severity_score'].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Number of Projects", f"{num_projects}")
kpi2.metric("Avg. Predicted Delay", f"{avg_delay:.0f} Days")
kpi3.metric("Avg. Predicted Overrun", f"{avg_cost_overrun:.2f} Lakhs")
kpi4.metric("Avg. Severity Score", f"{avg_severity:.0f} / 100")

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Severity Distribution")
    fig_pie = px.pie(vendor_df, names='project_type', values='severity_score', title=f"Severity Score by Project Type")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Delay vs. Cost Analysis")
    fig_scatter = px.scatter(vendor_df, x='predicted_delay', y='predicted_cost_overrun', color='project_type', size='severity_score', hover_data=['terrain'], title="Project Risk Profile (Size reflects Severity)")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Project Details")
st.dataframe(vendor_df[['project_type', 'terrain', 'predicted_delay', 'predicted_cost_overrun', 'severity_score']].sort_values('severity_score', ascending=False))