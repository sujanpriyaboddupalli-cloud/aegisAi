from typing import List, Dict

RECOMMENDATIONS: Dict[str, List[Dict]] = {
    "cooling_issue": [
        {"priority": "CRITICAL", "action": "Increase cooling fan speed to maximum.",
         "eta": "Immediate",  "impact": "Reduces temp ~8–12°C in 5 min"},
        {"priority": "HIGH",     "action": "Migrate workloads to cooler nodes.",
         "eta": "2–5 min",    "impact": "Reduces CPU load by 20–35%"},
        {"priority": "MEDIUM",   "action": "Inspect airflow obstructions in server rack.",
         "eta": "15 min",     "impact": "Long-term thermal stability"},
        {"priority": "LOW",      "action": "Schedule preventive cooling unit maintenance.",
         "eta": "Next window", "impact": "Prevents recurrence"},
    ],
    "power_failure": [
        {"priority": "CRITICAL", "action": "Switch to backup UPS immediately.",
         "eta": "Immediate",  "impact": "Prevents data loss from sudden shutdown"},
        {"priority": "HIGH",     "action": "Shed non-critical server loads.",
         "eta": "1–3 min",    "impact": "Reduces power draw by 15–25%"},
        {"priority": "MEDIUM",   "action": "Alert facilities team to check PDU circuits.",
         "eta": "5–10 min",   "impact": "Identifies root cause"},
    ],
    "server_overload": [
        {"priority": "CRITICAL", "action": "Throttle runaway processes immediately.",
         "eta": "Immediate",  "impact": "CPU relief of 10–30%"},
        {"priority": "HIGH",     "action": "Distribute jobs via load balancer.",
         "eta": "2–4 min",    "impact": "Even load distribution"},
        {"priority": "MEDIUM",   "action": "Enable auto-scaling if cloud burst available.",
         "eta": "3–8 min",    "impact": "Elastic capacity increase"},
    ],
    "network_saturation": [
        {"priority": "CRITICAL", "action": "Enable QoS — prioritise critical services.",
         "eta": "Immediate",  "impact": "Reduces latency for high-priority traffic"},
        {"priority": "HIGH",     "action": "Rate-limit non-essential egress traffic.",
         "eta": "2 min",      "impact": "Frees 20–40% bandwidth"},
        {"priority": "MEDIUM",   "action": "Investigate traffic for potential DDoS.",
         "eta": "5 min",      "impact": "Early threat detection"},
    ],
    "normal": [
        {"priority": "INFO",  "action": "No immediate action required.",
         "eta": "—",          "impact": "System is healthy"},
        {"priority": "LOW",   "action": "Continue scheduled monitoring and log archiving.",
         "eta": "Ongoing",    "impact": "Operational hygiene"},
    ],
}

PRIORITY_COLORS = {
    "CRITICAL": "#f87171",
    "HIGH":     "#fb923c",
    "MEDIUM":   "#fbbf24",
    "LOW":      "#34d399",
    "INFO":     "#60a5fa",
}

def get_recommendations(prediction: str) -> List[Dict]:
    return RECOMMENDATIONS.get(prediction, RECOMMENDATIONS["normal"])