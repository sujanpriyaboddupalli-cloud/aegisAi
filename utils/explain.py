import pandas as pd


def generate_explanation(row: pd.Series, prediction: str) -> str:
    cpu     = float(row.get("cpu",     0))
    temp    = float(row.get("temp",    0))
    power   = float(row.get("power",  0))
    network = float(row.get("network", 0))

    if prediction == "cooling_issue":
        return (
            f"<b>Cooling issue detected</b> — CPU is at <b>{cpu:.1f}%</b> and temperature has "
            f"climbed to <b>{temp:.1f}°C</b>. This combination indicates inadequate heat "
            f"dissipation and risks hardware degradation or thermal throttling. "
            f"Immediate cooling escalation and workload rebalancing are advised."
        )
    elif prediction == "power_failure":
        return (
            f"<b>Power failure risk</b> — Power draw is critically high at <b>{power:.1f}%</b>. "
            f"This exceeds safe operating thresholds and may cause circuit breaker trips "
            f"or UPS overload. Shift non-critical workloads and inspect feeder paths now."
        )
    elif prediction == "server_overload":
        return (
            f"<b>Server overload</b> — CPU has spiked to <b>{cpu:.1f}%</b> with sustained load. "
            f"Active processes are consuming more resources than the server can safely handle, "
            f"leading to latency degradation and potential service interruption."
        )
    elif prediction == "network_saturation":
        return (
            f"<b>Network saturation</b> — Network utilisation is at <b>{network:.1f}%</b>. "
            f"Bandwidth is near capacity, causing packet loss and increased latency "
            f"across connected services. Enable QoS and rate-limit non-critical egress."
        )
    else:
        return (
            f"<b>System nominal</b> — All metrics are within healthy thresholds. "
            f"CPU: <b>{cpu:.1f}%</b>, Temp: <b>{temp:.1f}°C</b>, "
            f"Power: <b>{power:.1f}%</b>, Network: <b>{network:.1f}%</b>. "
            f"Continue scheduled monitoring."
        )


def get_severity(prediction: str, anomaly: int, risk_score: float) -> str:
    if prediction != "normal" and anomaly == 1:
        return "critical"
    elif prediction != "normal" or (anomaly == 1 and risk_score > 0.6):
        return "warning"
    return "healthy"