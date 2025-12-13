# Upgraded_model_logic.py
# This file now handles BUSINESS RULES only (No ML training code here)

def generate_risk_recommendations(inputs, predicted_delay, predicted_cost, severity_score):
    """
    Translates ML numbers into actionable English advice.
    """
    recs = []

    # 1. Critical Severity
    if severity_score > 60:
        recs.append({
            "type": "Critical",
            "icon": "🚨",
            "msg": f"Risk Score {int(severity_score)}/100 is high. VP-level approval is required before proceeding."
        })

    # 2. Vendor Advice
    if inputs.get('vendor_rating', 3) < 3:
        # Calculate potential savings (Counterfactual logic)
        saved_days = int(predicted_delay * 0.20)
        recs.append({
            "type": "Actionable",
            "icon": "📉",
            "msg": f"Vendor Rating is low. Switching to a Tier-1 vendor could save ~{saved_days} days of delay."
        })

    # 3. Terrain Specifics
    terrain = inputs.get('terrain', '')
    if terrain == "Hilly":
        recs.append({
            "type": "Warning",
            "icon": "🏔️",
            "msg": "Hilly terrain risk: Allocate +15% budget for landslide contingencies and retaining walls."
        })
    elif terrain == "Urban":
        recs.append({
            "type": "Info",
            "icon": "🏙️",
            "msg": "Urban density risk: Start 'Right of Way' (RoW) clearances 3 months early to avoid bottlenecks."
        })

    # 4. Historical Context
    if inputs.get('historical_delays', 0) > 4:
        recs.append({
            "type": "Warning",
            "icon": "⚠️",
            "msg": "This region has a history of frequent delays. Weekly progress audits are recommended."
        })

    if not recs:
        recs.append({"type": "Success", "icon": "✅", "msg": "Project parameters look stable. Proceed with standard monitoring."})

    return recs