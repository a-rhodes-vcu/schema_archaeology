from models import Anomaly
from dataclasses import asdict
import datetime


def build_report(anomalies: list[Anomaly]) -> dict:
    """
    Assemble the final anomaly report dict from a list of Anomaly instances.

    Produces a structured report with:
        - A metadata header (timestamp, summary counts)
        - A sorted list of anomalies (critical first, then high, then medium)

    Parameters
    ----------
    anomalies : list[Anomaly]
        Output of run_checks() — one Anomaly per rule that found violations.

    Returns
    -------
    dict
        Report dict ready to be written to anomaly_report.json via json.dump().

    Design notes:
        Anomalies are sorted by severity so the most critical issues appear
        first in the report. asdict() converts each Anomaly dataclass to a
        plain dict for JSON serialisation.
    """

    # Sum all financial impacts across every anomaly for the report header
    total_impact = sum(a.estimated_financial_impact for a in anomalies)

    # Separate critical and high anomalies for the summary counts
    # List comprehensions filter by severity string
    critical = [a for a in anomalies if a.severity == "critical"]
    high     = [a for a in anomalies if a.severity == "high"]

    return {
        # UTC timestamp in ISO 8601 format with Z suffix (e.g. "2026-03-31T14:22:01Z")
        # utcnow() returns the current time in UTC — important for financial audit trails
        "report_generated_at": datetime.datetime.utcnow().isoformat() + "Z",

        # Summary section — high-level metrics for dashboard display
        "summary": {
            "total_anomalies_found": len(anomalies),

            # Count of critical-severity anomalies (AP-001, GL-001)
            "critical_count": len(critical),

            # Count of high-severity anomalies (PO-001, CR-001, DUP-001, AP-002)
            "high_count": len(high),

            # Total financial exposure across all anomalies, rounded to 2 decimal places
            # round() prevents floating-point artifacts like 12500.000000001
            "total_estimated_financial_exposure": round(total_impact, 2),

            # rules_evaluated == len(anomalies) because we only add an Anomaly
            # when a rule finds at least one violation. This could differ from
            # the total number of rules (8) if all rules pass cleanly.
            "rules_evaluated": len(anomalies),
        },

        # Full anomaly list, sorted by severity so critical issues appear first.
        # The sort key is a dict lookup that maps severity strings to integers:
        #   critical=0, high=1, medium=2, low=3
        # Lower numbers sort first, so critical anomalies bubble to the top.
        # asdict() converts each Anomaly dataclass instance to a plain dict
        # (required for json.dump() — json module cannot serialise dataclasses directly).
        "anomalies": [asdict(a) for a in sorted(
            anomalies,
            key=lambda a: {"critical": 0, "high": 1, "medium": 2, "low": 3}[a.severity]
        )]
    }
