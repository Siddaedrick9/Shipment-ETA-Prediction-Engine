"""Transparent stage-level delay propagation calculations."""

def propagate_delay(inherited_delay_hours, new_delay_hours, recovered_hours=0.0, schedule_slack_hours=0.0):
    """
    Propagate a delay through one stage.

    First consume schedule slack, then apply recovery, then add new delay.
    Negative inherited delay is treated as early/on-time for this simplified
    positive-delay propagation calculation.
    """
    inherited = max(0.0, float(inherited_delay_hours))
    slack = max(0.0, float(schedule_slack_hours))
    recovery = max(0.0, float(recovered_hours))
    new_delay = float(new_delay_hours)
    carried = max(0.0, inherited - slack - recovery)
    return carried + new_delay

def rule_based_eta_delay(features):
    """
    Baseline estimate at the port-arrival checkpoint.
    This is intentionally transparent and is not the trained ML prediction.
    """
    inherited = max(0.0, float(features["port_arrival_delay_hours"]))
    slack = max(0.0, float(features["schedule_slack_hours"]))
    congestion = float(features["port_congestion_index"])
    complexity = float(features["customs_complexity"])
    weather = float(features["weather_severity"])
    transshipments = float(features["transshipment_count"])

    port_pressure = max(0.0, congestion - 42.0) * 0.20
    customs_pressure = complexity * 1.45 + max(0.0, congestion - 65.0) * 0.10
    residual_weather = weather * 0.18
    transfer_risk = transshipments * 0.6
    recovery_allowance = 2.5
    return max(-8.0, max(0.0, inherited - slack) + port_pressure
               + customs_pressure + residual_weather + transfer_risk
               - recovery_allowance)
