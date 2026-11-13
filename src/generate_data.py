"""Generate a reproducible synthetic shipment dataset with stage-level timestamps."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_SHIPMENTS = 300

def generate_dataset(n=N_SHIPMENTS, seed=SEED):
    rng = np.random.default_rng(seed)
    origins = np.array(["Shanghai", "Singapore", "Rotterdam", "Mumbai", "Los Angeles"])
    destinations = np.array(["Rotterdam", "Hamburg", "Mumbai", "New York", "Singapore"])
    rows = []
    start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    for i in range(n):
        origin = str(rng.choice(origins))
        destination = str(rng.choice(destinations))
        while destination == origin:
            destination = str(rng.choice(destinations))
        distance = float(rng.integers(2500, 14500))
        planned_transit = distance / float(rng.uniform(25, 36))
        planned_port = float(rng.uniform(10, 26))
        planned_customs = float(rng.uniform(8, 30))
        planned_final_leg = float(rng.uniform(5, 22))
        scheduled_departure = start + pd.Timedelta(hours=int(rng.integers(0, 24*180)))
        scheduled_port_arrival = scheduled_departure + pd.Timedelta(hours=planned_transit)
        scheduled_customs = scheduled_port_arrival + pd.Timedelta(hours=planned_port)
        scheduled_delivery = scheduled_customs + pd.Timedelta(hours=planned_customs + planned_final_leg)

        congestion = float(np.clip(rng.beta(2.2, 2.0) * 100, 3, 99))
        weather = float(np.clip(rng.gamma(1.8, 1.8), 0, 10))
        customs_complexity = float(rng.integers(1, 11))
        transshipments = int(rng.choice([0, 1, 2, 3], p=[0.38, 0.40, 0.17, 0.05]))
        schedule_slack = float(rng.uniform(0, 18))

        departure_delay = float(np.clip(rng.normal(2 + congestion*0.025, 5), -5, 24))
        transit_weather_effect = weather * rng.uniform(0.45, 1.25)
        transit_delay = float(np.clip(rng.normal(transit_weather_effect + transshipments*1.8, 4), -8, 28))
        port_arrival_delay = departure_delay + transit_delay

        # Future downstream delay components are used only to create ground truth.
        port_extra = max(0.0, (congestion - 42.0) * 0.20) + rng.gamma(1.2, 1.1)
        customs_extra = customs_complexity * 1.45 + max(0.0, congestion - 65.0) * 0.10
        recovery = float(rng.uniform(0, 5.5))
        disruption = float(rng.normal(0, 4.0))
        inherited_after_slack = max(0.0, port_arrival_delay - schedule_slack)
        final_delay = inherited_after_slack + port_extra + customs_extra - recovery + disruption
        final_delay = float(np.clip(final_delay, -8, 100))

        actual_departure = scheduled_departure + pd.Timedelta(hours=departure_delay)
        actual_port_arrival = scheduled_port_arrival + pd.Timedelta(hours=port_arrival_delay)
        actual_customs = scheduled_customs + pd.Timedelta(hours=port_arrival_delay + port_extra)
        actual_delivery = scheduled_delivery + pd.Timedelta(hours=final_delay)

        rows.append({
            "shipment_id": f"SHP-{i+1:04d}",
            "origin_port": origin,
            "destination_port": destination,
            "distance_km": round(distance, 2),
            "scheduled_departure_utc": scheduled_departure.isoformat(),
            "scheduled_port_arrival_utc": scheduled_port_arrival.isoformat(),
            "scheduled_customs_utc": scheduled_customs.isoformat(),
            "scheduled_delivery_utc": scheduled_delivery.isoformat(),
            "actual_departure_utc": actual_departure.isoformat(),
            "actual_port_arrival_utc": actual_port_arrival.isoformat(),
            "actual_customs_utc": actual_customs.isoformat(),
            "actual_delivery_utc": actual_delivery.isoformat(),
            "departure_delay_hours": round(departure_delay, 3),
            "transit_delay_hours": round(transit_delay, 3),
            "port_arrival_delay_hours": round(port_arrival_delay, 3),
            "port_congestion_index": round(congestion, 2),
            "weather_severity": round(weather, 2),
            "customs_complexity": customs_complexity,
            "planned_customs_hours": round(planned_customs, 2),
            "schedule_slack_hours": round(schedule_slack, 2),
            "transshipment_count": transshipments,
            "actual_port_extra_hours": round(port_extra, 3),
            "actual_customs_extra_hours": round(customs_extra, 3),
            "recovered_delay_hours": round(recovery, 3),
            "actual_delivery_delay_hours": round(final_delay, 3),
            "is_delayed_24h": int(final_delay > 24),
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic_shipments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} shipments to {out}")
