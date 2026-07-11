"""
models.py — Shared data models for the P2P context engine.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Anomaly:
    rule_id:                    str
    severity:                   str   # critical | high | medium | low
    category:                   str   # AP_control | 3way_match | GL_integrity | credit_risk | duplicate
    description:                str
    affected_records:           list[dict[str, Any]] = field(default_factory=list)
    count:                      int   = 0
    estimated_financial_impact: float = 0.0
    remediation:                str   = ""