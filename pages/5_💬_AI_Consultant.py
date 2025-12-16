import streamlit as st
from groq import Groq
import pandas as pd
import json

st.set_page_config(page_title="AI Consultant", page_icon="💬", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #e6f3ff; border-left: 5px solid #0083b8;
    }
    .chat-message.bot {
        background-color: #f0f2f6; border-left: 5px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. SECURE API KEY LOAD ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    st.error("🚨 API Key not found! Please set GROQ_API_KEY in .streamlit/secrets.toml")
    st.stop()

# --- 2. INTELLIGENT CONTEXT LOADING ---
# This logic checks if the user has run a simulation in the other tab.
# If "simulation_input" exists in session_state, we use THAT as the context.
if "simulation_input" in st.session_state:
    current_context = st.session_state.simulation_input
    source_msg = "🟢 Linked to Live Simulation Data"
else:
    # Default Fallback (If user goes straight to Chatbot without simulating)
    current_context = {
        "project_type": "Overhead Line (Default)",
        "terrain": "Hilly",
        "material_cost": "₹200 Cr",
        "vendor_rating": "3/5 (Average)",
        "risk_status": "High Risk (Demo)"
    }
    source_msg = "🟡 Using Demo Context (Run a simulation to update)"

# --- SIDEBAR DISPLAY (Beautified) ---
st.sidebar.title("⚙️ Mission Control")
st.sidebar.success(source_msg)

st.sidebar.markdown("### 🏗️ Active Project Intel")

# We use an Expander so it doesn't clutter the screen
with st.sidebar.expander("📝 View Parameters", expanded=True):
    # Iterate and make it look clean
    for key, value in current_context.items():
        # Clean up the key name (e.g. "project_type" -> "Project Type")
        label = key.replace("_", " ").title()
        
        # Display as small metrics
        st.markdown(f"**{label}:**") 
        st.info(f"{value}")

# Add a "Clear Context" button for safety
if st.sidebar.button("🔄 Reset Context"):
    if "simulation_input" in st.session_state:
        del st.session_state.simulation_input
    st.rerun()

# --- 3. MAIN CHAT INTERFACE ---
st.title("💬 Infralytics AI Consultant")
st.markdown("##### Your 24/7 Infrastructure Risk Expert")
st.caption("This AI uses **Llama-3-70b** (via Groq) to provide engineering-grade risk mitigation strategies.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have access to your current project parameters. Based on the data, what specific risk would you like me to analyze?"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. STRICT CHAT LOGIC ---
if prompt := st.chat_input("Ex: 'How can I mitigate the delay in Hilly terrain?'"):
    
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- THE BRAIN: ENGINEERING SYSTEM PROMPT ---
    context_str = json.dumps(current_context, indent=2)
    
    system_prompt = f"""
    ROLE: 
    You are the Chief Risk Engineer for 'Infralytics'. Your job is to minimize delays and cost overruns.
    
    STRICT DATA CONTEXT:
    {context_str}
    
    INSTRUCTIONS:
    1.  **USE THE DATA:** If the user asks about risk, refer strictly to the 'terrain', 'vendor_rating', or 'cost' provided in the context above.
    2.  **BE PRECISE:** Do not give generic advice. If the terrain is 'Hilly', mention landslides, soil erosion, or logistics. If 'Coastal', mention corrosion or cyclones.
    3.  **GUARDRAILS:** If the user asks non-engineering questions (like "tell me a joke" or "politics"), politely refuse and steer back to infrastructure.
    4.  **FORMAT:** Use bullet points for solutions. Keep answers under 150 words.
    
    TONE: Professional, Technical, Direct.
    """

    try:
        client = Groq(api_key=api_key)
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.messages
            ],
            stream=True,
            temperature=0.3, # <--- LOWER TEMP = LESS RANDOM, MORE FACTUAL
            max_tokens=400   # <--- Keep answers concise
        )

        # Helper to parse stream
        def parse_groq_stream(stream):
            for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

        with st.chat_message("assistant"):
            response = st.write_stream(parse_groq_stream(stream))
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"AI Connection Error: {str(e)}")