"""Streamlit dashboard for shipment ETA and delay-propagation intelligence."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.propagation import rule_based_eta_delay

st.set_page_config(page_title="Shipment ETA Intelligence", page_icon="🚢", layout="wide")
st.title("🚢 Shipment ETA & Delay Intelligence")
st.caption("Checkpoint-based ETA prediction, delay propagation, risk scoring and scenario analysis")

@st.cache_resource
def load_assets():
    reg = joblib.load(ROOT / "models" / "eta_regressor.joblib")
    clf = joblib.load(ROOT / "models" / "delay_classifier.joblib")
    metrics = json.loads((ROOT / "models" / "model_metrics.json").read_text())
    data = pd.read_csv(ROOT / "data" / "synthetic_shipments.csv")
    return reg, clf, metrics, data

regressor, classifier, metrics, data = load_assets()
features = metrics["features"]

with st.sidebar:
    st.header("Shipment checkpoint")
    st.write("Prediction is made immediately after port arrival.")
    shipment_ids = data["shipment_id"].tolist()
    selected_id = st.selectbox("Load a synthetic shipment", shipment_ids, index=0)
    row = data.loc[data.shipment_id == selected_id].iloc[0]
    st.divider()
    st.subheader("Known operational conditions")
    distance = st.number_input("Route distance (km)", 500.0, 20000.0, float(row.distance_km), 100.0)
    departure_delay = st.number_input("Departure delay (hours)", -12.0, 72.0, float(row.departure_delay_hours), 1.0)
    transit_delay = st.number_input("Transit delay (hours)", -24.0, 72.0, float(row.transit_delay_hours), 1.0)
    congestion = st.slider("Port congestion index", 0, 100, int(row.port_congestion_index))
    weather = st.slider("Weather severity", 0.0, 10.0, float(row.weather_severity), 0.1)
    complexity = st.slider("Customs complexity", 1, 10, int(row.customs_complexity))
    customs_hours = st.slider("Planned customs duration (hours)", 4.0, 60.0, float(row.planned_customs_hours), 1.0)
    slack = st.slider("Available schedule slack (hours)", 0.0, 36.0, float(row.schedule_slack_hours), 1.0)
    transfers = st.slider("Transshipment count", 0, 5, int(row.transshipment_count))

x = pd.DataFrame([{
    "distance_km": distance,
    "departure_delay_hours": departure_delay,
    "transit_delay_hours": transit_delay,
    "port_arrival_delay_hours": departure_delay + transit_delay,
    "port_congestion_index": congestion,
    "weather_severity": weather,
    "customs_complexity": complexity,
    "planned_customs_hours": customs_hours,
    "schedule_slack_hours": slack,
    "transshipment_count": transfers
}])[features]

predicted_delay = float(regressor.predict(x)[0])
risk_probability = float(classifier.predict_proba(x)[0, 1])
rule_delay = float(rule_based_eta_delay(x.iloc[0].to_dict()))
scheduled_delivery = pd.to_datetime(row.scheduled_delivery_utc, utc=True)
predicted_eta = scheduled_delivery + pd.Timedelta(hours=predicted_delay)
actual_eta = pd.to_datetime(row.actual_delivery_utc, utc=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted delivery delay", f"{predicted_delay:+.1f} h")
c2.metric("Delay risk (>24h)", f"{risk_probability:.0%}")
c3.metric("Predicted ETA (UTC)", predicted_eta.strftime("%d %b, %H:%M"))
c4.metric("Transparent rule baseline", f"{rule_delay:+.1f} h")

st.subheader("Shipment journey")
st.caption(f"Selected synthetic record: {selected_id} · {row.origin_port} → {row.destination_port}")
timeline = pd.DataFrame({
    "Stage": ["Scheduled departure", "Actual departure", "Scheduled port arrival", "Actual port arrival", "Scheduled delivery", "Predicted delivery", "Actual delivery (evaluation only)"],
    "Timestamp (UTC)": [
        pd.to_datetime(row.scheduled_departure_utc, utc=True),
        pd.to_datetime(row.actual_departure_utc, utc=True),
        pd.to_datetime(row.scheduled_port_arrival_utc, utc=True),
        pd.to_datetime(row.actual_port_arrival_utc, utc=True),
        scheduled_delivery, predicted_eta, actual_eta
    ]
})
st.dataframe(timeline, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Delay propagation trace")
    inherited = max(0.0, float(x.iloc[0]["port_arrival_delay_hours"]))
    consumed = min(inherited, float(slack))
    carried = max(0.0, inherited - float(slack))
    port_pressure = max(0.0, (float(congestion) - 42.0) * 0.20)
    customs_pressure = float(complexity) * 1.45 + max(0.0, float(congestion)-65.0)*0.10
    trace = pd.DataFrame({
        "Component": ["Inherited delay at port", "Schedule slack absorbed", "Delay carried forward", "Congestion-related port pressure", "Customs complexity pressure"],
        "Hours": [inherited, -consumed, carried, port_pressure, customs_pressure]
    })
    st.dataframe(trace, use_container_width=True, hide_index=True)
    st.caption("Operational trace is a transparent approximation; the ML estimate also learns residual variation from synthetic training journeys.")

with right:
    st.subheader("Top model drivers")
    try:
        import shap
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(x)
        vals = np.asarray(shap_values).reshape(-1)
        importance = pd.DataFrame({"Feature": features, "Contribution": vals})
        importance["Absolute contribution"] = importance["Contribution"].abs()
        importance = importance.sort_values("Absolute contribution", ascending=False).head(7)
        st.bar_chart(importance.set_index("Feature")["Contribution"])
        st.caption("Local SHAP contributions relative to the model baseline.")
    except Exception:
        if hasattr(regressor, "feature_importances_"):
            importance = pd.Series(regressor.feature_importances_, index=features).sort_values(ascending=False).head(7)
            st.bar_chart(importance)
            st.caption("Global tree feature importance fallback. This is not a per-shipment causal explanation.")
        else:
            st.info("Install SHAP to enable local feature-attribution visualization.")

st.divider()
st.subheader("What-if scenario simulator")
st.write("Change future operating conditions and compare the model's ETA estimate with the current scenario.")
s1, s2, s3 = st.columns(3)
with s1:
    scenario_congestion = st.slider("Scenario congestion", 0, 100, max(0, congestion-20), key="scenario_cong")
with s2:
    scenario_customs = st.slider("Scenario customs complexity", 1, 10, max(1, complexity-2), key="scenario_customs")
with s3:
    scenario_weather = st.slider("Scenario weather severity", 0.0, 10.0, max(0.0, weather-1.5), 0.1, key="scenario_weather")
scenario = x.copy()
scenario.loc[0, "port_congestion_index"] = scenario_congestion
scenario.loc[0, "customs_complexity"] = scenario_customs
scenario.loc[0, "weather_severity"] = scenario_weather
scenario_delay = float(regressor.predict(scenario)[0])
delta = predicted_delay - scenario_delay
m1, m2 = st.columns(2)
m1.metric("Scenario predicted delay", f"{scenario_delay:+.1f} h")
m2.metric("Estimated improvement", f"{delta:+.1f} h", help="Positive means the scenario predicts fewer delay hours.")
st.caption("Scenario results are model-based estimates, not guaranteed operational savings.")

with st.expander("Model evaluation and dataset details"):
    st.write(f"Training dataset: {metrics['n_samples']} synthetic shipments. Held-out test size: {metrics['test_samples']}.")
    st.write(f"Selected regression model: **{metrics['selected_model']}**")
    st.dataframe(pd.DataFrame(metrics["regression_results"]).T, use_container_width=True)
    st.json(metrics["classification_metrics"])
    st.warning("This project uses synthetic data for demonstration. Metrics do not establish real-world shipping performance.")
