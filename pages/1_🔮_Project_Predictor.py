import streamlit as st
import pandas as pd
import plotly.express as px

# --- IMPORT THE NEW AI ENGINE ---
# We no longer load models here. We just ask the engine for a prediction.
from ml.predict import get_prediction
from upgraded_model_logic import generate_risk_recommendations

st.set_page_config(page_title="Infralytics | Simulator", page_icon="🔮", layout="wide")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .rec-box {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 15px;
        display: flex;
        align-items: center;
        border-left-width: 5px;
        border-left-style: solid;
    }
    .rec-critical { background-color: #ffe6e6; border-left-color: #ff4b4b; }
    .rec-warning { background-color: #fff4e5; border-left-color: #ffbd45; }
    .rec-actionable { background-color: #e6f3ff; border-left-color: #0083b8; }
    .rec-success { background-color: #e6fffa; border-left-color: #00cc96; }
    .rec-financial { background-color: #f3e5f5; border-left-color: #9c27b0; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🔮 AI Project Risk Simulator")
st.markdown("""
**Interactive Scenario Planning:** Adjust project parameters below to see how they impact 
timeline delays and cost overruns in real-time using our **Counterfactual AI Engine**.
""")
st.divider()

# --- INPUT SECTION (SIMULATOR CONTROLS) ---
col_controls, col_dashboard = st.columns([1, 2])

with col_controls:
    st.subheader("🎛️ Simulation Parameters")
    with st.container(border=True):
        # Input Controls
        project_type = st.selectbox("Project Type", ["Overhead Line", "Substation", "Underground Cable"])
        terrain = st.selectbox("Terrain", ["Plain", "Urban", "Hilly", "Coastal"])
        
        st.markdown("---")
        st.caption("Risk Factors")
        vendor_rating = st.slider("Vendor Rating (1=Poor, 5=Best)", 1, 5, 3)
        hist_delays = st.slider("Historical Regional Delays", 0, 15, 2)
        
        st.markdown("---")
        st.caption("Financials")
        mat_cost = st.number_input("Material Cost (₹ Cr)", 50, 1000, 200, step=10)
        
        # Hidden Defaults (Can be made dynamic later)
        lat, lon = 28.6, 77.2
        
        run_sim = st.button("🚀 Run AI Simulation", type="primary", use_container_width=True)

# --- DASHBOARD SECTION ---
with col_dashboard:
    if run_sim:
        # Prepare Input Data
        input_data = {
            "project_type": project_type,
            "terrain": terrain,
            "material_cost_crore": mat_cost,
            "vendor_performance_rating": vendor_rating,
            "historical_delays_project_count": hist_delays,
            "lat": lat, "lon": lon
        }

        # 1. CALL THE INFERENCE ENGINE
        # This returns everything we need: predictions, severity, AND feature names
        results = get_prediction(input_data)
        
        delay = results['predicted_delay']
        cost = results['predicted_cost_overrun']
        score = results['severity_score']
        feature_names = results['feature_names'] # Extracted from the result

        # 2. DISPLAY METRICS
        st.subheader("📊 Projected Outcomes")
        
        m1, m2, m3 = st.columns(3)
        
        # Color Logic for Severity
        sev_color = "normal"
        if score > 60: sev_color = "inverse" # Red if high risk
        
        m1.metric("Predicted Delay", f"{int(delay)} Days", delta="AI Estimate", delta_color="off")
        m2.metric("Cost Overrun", f"₹ {int(cost)} Lakhs", delta="Risk Factor", delta_color="inverse")
        m3.metric("Severity Score", f"{int(score)} / 100", delta=f"{'High Risk' if score>60 else 'Stable'}", delta_color=sev_color)
        
        st.markdown("---")

        # 3. RECOMMENDATION ENGINE
        st.subheader("🧠 AI Risk Mitigation Strategy")
        
        recs = generate_risk_recommendations(input_data, delay, cost, score)
        
        for rec in recs:
            # Map type to CSS class
            css_map = {
                "Critical": "rec-critical",
                "Warning": "rec-warning", 
                "Actionable": "rec-actionable",
                "Success": "rec-success",
                "Financial": "rec-financial"
            }
            css_class = css_map.get(rec['type'], "rec-actionable")
            
            st.markdown(f"""
            <div class="rec-box {css_class}">
                <span style="font-size:20px; margin-right:10px;">{rec.get('icon', '💡')}</span>
                <div>
                    <strong>{rec['type']}</strong><br>
                    {rec['msg']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # 4. EXPLAINABILITY (SHAP CHART)
        st.subheader("🔎 What is driving this risk?")
        st.caption("SHAP Value Analysis: Features pushing risk up (Red) or down (Blue)")
        
        shap_vals = results['shap_values'][0] # Get array
        
        # Create DataFrame for Plotly
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Impact': shap_vals
        })
        
        # Add color logic
        feature_importance['Type'] = feature_importance['Impact'] > 0
        feature_importance['Abs_Impact'] = feature_importance['Impact'].abs()
        
        # Sort and take top 7
        top_features = feature_importance.sort_values('Abs_Impact', ascending=False).head(7)
        
        # Plot
        fig = px.bar(
            top_features, 
            x="Impact", 
            y="Feature", 
            orientation='h',
            color="Type",
            color_discrete_map={True: '#ff4b4b', False: '#0083b8'}, # Red for bad, Blue for good
            title="Top Factors Influencing Delay"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Empty State
        st.info("👈 Adjust parameters on the left and click 'Run Simulation' to generate a risk profile.")
        
        # Placeholder visual
        st.markdown("""
        <div style="text-align: center; color: #888; padding: 50px;">
            <h3>Waiting for Input...</h3>
            <p>Select Project Type and Terrain to begin.</p>
        </div>
        """, unsafe_allow_html=True)