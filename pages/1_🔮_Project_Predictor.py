import streamlit as st
import pandas as pd
import plotly.express as px

# --- IMPORT ENGINES ---
from ml.predict import get_prediction
from upgraded_model_logic import generate_risk_recommendations
from utils import generate_pdf_report

st.set_page_config(page_title="Infralytics | Simulator", page_icon="🔮", layout="wide")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Recommendation Boxes */
    .rec-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 15px;
        display: flex;
        align-items: flex-start;
        border-left-width: 5px;
        border-left-style: solid;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .rec-critical { background-color: #fff1f2; border-left-color: #e11d48; }
    .rec-warning { background-color: #fffbeb; border-left-color: #f59e0b; }
    .rec-actionable { background-color: #eff6ff; border-left-color: #3b82f6; }
    .rec-success { background-color: #f0fdf4; border-left-color: #22c55e; }
    
    /* Optimizer Result Card */
    .opt-card { 
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #bae6fd; 
        margin-bottom: 25px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🔮 AI Project Risk Simulator")
st.markdown("""
**Interactive Scenario Planning:** Adjust project parameters to predict outcomes. 
Use the **✨ Auto-Optimize** button to let AI find the most efficient setup.
""")
st.divider()

# --- INPUT SECTION ---
col_controls, col_dashboard = st.columns([1, 2])

with col_controls:
    st.subheader("🎛️ Parameters")
    with st.container(border=True):
        project_type = st.selectbox("Project Type", ["Overhead Line", "Substation", "Underground Cable"])
        terrain = st.selectbox("Terrain", ["Plain", "Urban", "Hilly", "Coastal"])
        
        st.markdown("---")
        st.caption("Risk Factors")
        vendor_rating = st.slider("Vendor Rating (1=Poor, 5=Best)", 1, 5, 3)
        hist_delays = st.slider("Regional Delay History", 0, 15, 2)
        
        st.markdown("---")
        st.caption("Financials")
        mat_cost = st.number_input("Budget (₹ Cr)", 50, 1000, 200, step=10)
        
        # Hidden Defaults
        lat, lon = 28.6, 77.2
        
        # ACTION BUTTONS
        col_btn1, col_btn2 = st.columns(2)
        run_sim = col_btn1.button("🚀 Simulate", type="primary", use_container_width=True)
        run_opt = col_btn2.button("✨ Optimize", help="AI will find the best configuration", use_container_width=True)

# --- MAIN LOGIC ---
if run_sim or run_opt:
    with col_dashboard:
        # 1. BASELINE PREDICTION
        input_data = {
            "project_type": project_type,
            "terrain": terrain,
            "material_cost_crore": mat_cost,
            "vendor_performance_rating": vendor_rating,
            "historical_delays_project_count": hist_delays,
            "lat": lat, "lon": lon
        }
        
        # Get Predictions
        res = get_prediction(input_data)
        delay = res['predicted_delay']
        cost = res['predicted_cost_overrun']
        score = res['severity_score']
        feature_names = res['feature_names']
        shap_values = res['shap_values']

        # Save to Session State (For Chatbot)
        st.session_state.simulation_input = {
            "project_type": project_type, "terrain": terrain,
            "material_cost": f"₹{mat_cost}Cr", "vendor_rating": f"{vendor_rating}/5",
            "predicted_delay": f"{int(delay)} Days", "predicted_cost_overrun": f"₹{int(cost)}L",
            "risk_score": f"{int(score)}"
        }

        # --- OPTIMIZER LOGIC (Triggered if 'Optimize' clicked) ---
        if run_opt:
            st.subheader("✨ AI Optimization Result")
            
            # Scenario: What if we force max vendor rating?
            opt_input = input_data.copy()
            opt_input['vendor_performance_rating'] = 5 
            
            opt_res = get_prediction(opt_input)
            
            saved_days = delay - opt_res['predicted_delay']
            saved_cost = cost - opt_res['predicted_cost_overrun']
            
            if saved_days > 5:
                st.markdown(f"""
                <div class="opt-card">
                    <h4 style="margin:0; color:#0369a1;">🚀 Optimization Opportunity Found!</h4>
                    <p style="margin-bottom:15px;">By upgrading to a <strong>Tier-1 Vendor (Rating 5)</strong>, our AI projects:</p>
                    <div style="display:flex; justify-content:space-around;">
                        <div style="text-align:center;">
                            <div style="font-size:24px; font-weight:bold; color:#15803d;">-{int(saved_days)} Days</div>
                            <div style="font-size:12px; color:#666;">Faster Completion</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:24px; font-weight:bold; color:#15803d;">₹ -{int(saved_cost)} L</div>
                            <div style="font-size:12px; color:#666;">Cost Savings</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.info("✅ This project configuration is already optimal for the given terrain.")
            
            st.markdown("---")

        # 2. METRICS DASHBOARD
        st.subheader("📊 Projected Outcomes")
        m1, m2, m3 = st.columns(3)
        sev_col = "inverse" if score > 60 else "normal"
        
        m1.metric("Predicted Delay", f"{int(delay)} Days", delta="AI Estimate", delta_color="off")
        m2.metric("Cost Overrun", f"₹ {int(cost)} Lakhs", delta="Risk Factor", delta_color="inverse")
        m3.metric("Severity Score", f"{int(score)} / 100", delta="Risk Level", delta_color=sev_col)
        
        st.markdown("---")

        # 3. RECOMMENDATIONS (Restored Full Styling)
        st.subheader("🧠 Strategy & Recommendations")
        recs = generate_risk_recommendations(input_data, delay, cost, score)
        
        for rec in recs:
            # Map type to CSS
            css_map = {
                "Critical": "rec-critical",
                "Warning": "rec-warning", 
                "Actionable": "rec-actionable",
                "Success": "rec-success"
            }
            css_class = css_map.get(rec['type'], "rec-actionable")
            icon = rec.get('icon', '💡')
            
            st.markdown(f"""
            <div class="rec-box {css_class}">
                <div style="font-size:24px; margin-right:15px;">{icon}</div>
                <div>
                    <div style="font-weight:bold; margin-bottom:4px;">{rec['type']}</div>
                    {rec['msg']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # 4. EXPLAINABILITY (Restored SHAP Chart)
        st.subheader("🔎 Risk Drivers (Explainable AI)")
        st.caption("These factors are contributing most to the predicted delay.")
        
        if shap_values is not None:
            shap_vals = shap_values[0] # Get array
            
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Impact': shap_vals
            })
            feature_importance['Type'] = feature_importance['Impact'] > 0
            feature_importance['Abs_Impact'] = feature_importance['Impact'].abs()
            top_features = feature_importance.sort_values('Abs_Impact', ascending=False).head(7)
            
            fig = px.bar(
                top_features, x="Impact", y="Feature", orientation='h',
                color="Type", color_discrete_map={True: '#ff4b4b', False: '#0083b8'},
                title="Top Factors Influencing Delay"
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 5. REPORT EXPORT (Fail-safe Method)
        st.subheader("📄 Export Analysis")
        st.caption("Download a professional PDF audit report for this simulation scenario.")
        
        pdf_preds = {"delay": delay, "cost": cost, "score": score}
        
        try:
            pdf_bytes = generate_pdf_report(input_data, pdf_preds, recs)
            st.download_button(
                label="📥 Download Audit Report (PDF)",
                data=pdf_bytes,
                file_name="Infralytics_Risk_Audit.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="download_btn"
            )
        except Exception as e:
            st.error(f"⚠️ PDF Error: {e}")

else:
    # Empty State
    with col_dashboard:
        st.info("👈 Select parameters and click **Simulate** or **Optimize** to begin.")
        st.markdown("""
        <div style="text-align: center; color: #888; padding: 50px;">
            <h3>Waiting for Input...</h3>
            <p>Select Project Type and Terrain to begin.</p>
        </div>
        """, unsafe_allow_html=True)