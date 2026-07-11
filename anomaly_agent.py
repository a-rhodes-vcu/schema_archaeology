# increases anomaly detection agent for an ever increasing changing data and data changing over time
# tables,columns stay the same - set mastered ententies change. need for data enrichment, grow over time
# graph data base with customser info. data enrichment table, pull from records and find anomalies.
# data shouldn't grow boundless

"""
anomaly_agent.py — Data Integrity Agent for P2P.

Purpose:
    Scans the P2P SQLite database for financial anomalies using 8 SQL-based
    rules and outputs a structured JSON report. Each anomaly includes a severity
    rating, affected records, estimated financial impact, and remediation steps.

    Goes beyond the minimum spec to catch edge cases like inactive vendor
    invoices and workflow-stalled pending invoices past their due date.

    Anomaly categories covered:
        GL_integrity  — missing GL entries, imbalanced debits/credits
        3way_match    — missing goods receipts, over-received POs
        credit_risk   — vendors exceeding their credit limit
        duplicate     — duplicate invoice numbers per vendor
        AP_control    — workflow stalls, inactive vendor invoices

Usage:
    python anomaly_agent.py --db p2p.db --out anomaly_report.json
"""


from logger import get_logger
log = get_logger("anomaly_agent")

from models import Anomaly
from dataclasses import dataclass, asdict, field
from typing import Any
import sqlite3, json, argparse, datetime



# ── run_checks ────────────────────────────────────────────────────────────────

def run_checks(db_path: str) -> list[Anomaly]:
    """
    Connect to the P2P database and run all 8 anomaly detection rules.

    Each rule executes one SQL query, wraps any results into an Anomaly
    dataclass instance, and appends it to the anomalies list. Rules only
    produce an Anomaly if they find at least one affected record.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (e.g. "p2p.db").

    Returns
    -------
    list[Anomaly]
        List of Anomaly instances, one per rule that found violations.
        Empty list if the database is clean.

    Design notes:
        - All rules run in a single database connection (one open/close cycle)
        - affected_records is capped at 20 rows to keep the JSON report readable
        - Financial impact is always the sum of relevant amounts, not a sample
        - Rules are ordered by severity: critical first, then high, then medium
    """

    con = sqlite3.connect(db_path)

    # sqlite3.Row makes each row behave like a dict — you can access columns
    # by name (r["amount"]) instead of by index (r[0]). Essential for building
    # the affected_records dicts and for the financial impact sum expressions.
    con.row_factory = sqlite3.Row

    cur = con.cursor()

    # Initialise the empty list that will accumulate Anomaly instances
    anomalies: list[Anomaly] = []

    # ── RULE 1: AP-001 — Invoices approved without GL entry ──────────────────
    # Severity: critical
    # Category: GL_integrity
    #
    # An invoice that is approved or paid should always have at least one GL entry.
    # Without a GL entry, the financial impact is unrecorded — the invoice shows
    # on the P&L but has no corresponding accounting entry. This is a critical
    # control failure that must be resolved before period close.
    #
    # SQL logic:
    #   - Select invoices with status 'approved' or 'paid'
    #   - Exclude invoices whose id appears in gl_entries (using NOT IN subquery)
    #   - Join vendors to include vendor name in the affected records
    rows = cur.execute("""
        SELECT i.id, i.invoice_number, i.amount, i.vendor_id, v.name AS vendor_name
        FROM invoices i
        JOIN vendors v ON v.id = i.vendor_id
        WHERE i.status IN ('approved', 'paid')
          AND i.id NOT IN (SELECT DISTINCT invoice_id FROM gl_entries)
    """).fetchall()

    # Only create an Anomaly if at least one affected row was found
    if rows:
        a = Anomaly(
            rule_id="AP-001",
            severity="critical",
            category="GL_integrity",
            description="Invoices approved/paid with no corresponding GL entry",
            # Cap affected_records at 20 rows — enough for review without bloating the report
            # dict(r) converts each sqlite3.Row to a plain dict for JSON serialisation
            affected_records=[dict(r) for r in rows[:20]],
            # count reflects the full number of violations, even if records are capped
            count=len(rows),
            # Sum all invoice amounts — the total unrecorded financial exposure
            estimated_financial_impact=sum(r["amount"] for r in rows),
            remediation="Finance team must post GL entries or reverse approval status"
        )
        # Add this anomaly to the running list
        anomalies.append(a)

    # ── RULE 2: PO-001 — POs with qty received > qty ordered ─────────────────
    # Severity: high
    # Category: 3way_match
    #
    # A purchase order line item's received quantity should never exceed the
    # ordered quantity without an approved amendment. Over-receipt can indicate:
    #   - Warehouse error (receiving too many units)
    #   - Data entry error (wrong PO referenced)
    #   - Intentional fraud (receiving goods not authorised)
    #
    # Financial impact is the overrun value: (qty_received - qty_ordered) × unit_cost
    # This represents the value of goods received without authorisation.
    #
    # SQL logic:
    #   - Select line items where qty_received > qty_ordered
    #   - Calculate overrun_value inline as a derived column
    rows = cur.execute("""
        SELECT li.po_id, li.sku, li.qty_ordered, li.qty_received,
               li.unit_cost,
               (li.qty_received - li.qty_ordered) * li.unit_cost AS overrun_value
        FROM po_line_items li
        WHERE li.qty_received > li.qty_ordered
    """).fetchall()

    if rows:
        a = Anomaly(
            rule_id="PO-001",
            severity="high",
            category="3way_match",
            description="Purchase order line items where received quantity exceeds ordered quantity",
            affected_records=[dict(r) for r in rows[:20]],
            count=len(rows),
            # Sum the overrun_value for each line — total unauthorised receipt value
            estimated_financial_impact=sum(r["overrun_value"] for r in rows),
            remediation="Raise amendment PO or return excess goods; update line items"
        )
        anomalies.append(a)

    # ── RULE 3: CR-001 — Vendors over their credit limit ─────────────────────
    # Severity: high
    # Category: credit_risk
    #
    # Each vendor has a credit_limit — the maximum outstanding payable balance
    # the company will carry for that vendor. If open invoices exceed this limit,
    # the company is exposed to more credit risk than approved.
    #
    # "Open" invoices = all statuses except 'paid' and 'cancelled'.
    # We exclude paid invoices (settled, no longer a liability) and cancelled
    # invoices (never valid).
    #
    # Financial impact is the excess above the credit limit — how much over
    # the approved threshold each vendor sits.
    #
    # SQL logic:
    #   - Group invoices by vendor, sum open invoice amounts
    #   - HAVING filters to only vendors where the sum exceeds their credit_limit
    #   - ORDER BY excess DESC to surface the worst offenders first
    rows = cur.execute("""
        SELECT v.id AS vendor_id, v.name, v.credit_limit,
               SUM(i.amount) AS total_invoiced,
               SUM(i.amount) - v.credit_limit AS excess
        FROM vendors v
        JOIN invoices i ON i.vendor_id = v.id
        WHERE i.status NOT IN ('paid', 'cancelled')
        GROUP BY v.id, v.name, v.credit_limit
        HAVING total_invoiced > v.credit_limit
        ORDER BY excess DESC
    """).fetchall()

    if rows:
        a = Anomaly(
            rule_id="CR-001",
            severity="high",
            category="credit_risk",
            description="Vendors with open invoice exposure exceeding their credit limit",
            affected_records=[dict(r) for r in rows[:20]],
            count=len(rows),
            # Sum the excess amounts — total credit headroom breach across all vendors
            estimated_financial_impact=sum(r["excess"] for r in rows),
            remediation="AP team to review and escalate; no new POs until credit headroom restored"
        )
        anomalies.append(a)

    # ── RULE 4: GL-001 — GL entries that don't balance per invoice ────────────
    # Severity: critical
    # Category: GL_integrity
    #
    # Double-entry accounting requires that for every invoice, the sum of debits
    # must equal the sum of credits in gl_entries. Any imbalance is a fundamental
    # accounting error that will cause the trial balance to be out of balance.
    #
    # We use ABS() to catch both over-debited and over-credited entries.
    # The threshold is 0.01 (one cent) to avoid flagging floating-point rounding
    # noise from financial calculations.
    #
    # SQL logic:
    #   - Group gl_entries by invoice_id, sum debits and credits separately
    #   - HAVING filters to only invoices where ABS(debit - credit) > 0.01
    #   - imbalance = the absolute difference between debits and credits
    rows = cur.execute("""
        SELECT invoice_id,
               SUM(debit)  AS total_debit,
               SUM(credit) AS total_credit,
               ABS(SUM(debit) - SUM(credit)) AS imbalance
        FROM gl_entries
        GROUP BY invoice_id
        HAVING ABS(SUM(debit) - SUM(credit)) > 0.01
        ORDER BY imbalance DESC
    """).fetchall()

    if rows:
        a = Anomaly(
            rule_id="GL-001",
            severity="critical",
            category="GL_integrity",
            description="GL entries where debits ≠ credits per invoice (double-entry violation)",
            affected_records=[dict(r) for r in rows[:20]],
            count=len(rows),
            # Sum of all imbalances — total accounting discrepancy across all affected invoices
            estimated_financial_impact=sum(r["imbalance"] for r in rows),
            remediation="Journal entry correction required; flag for month-end close review"
        )
        anomalies.append(a)

    # ── RULE 5: DUP-001 — Duplicate invoice numbers from same vendor ──────────
    # Severity: high
    # Category: duplicate
    #
    # Each vendor should issue unique invoice numbers. A duplicate invoice number
    # from the same vendor means either:
    #   - The vendor accidentally reused a number (common billing system error)
    #   - The same invoice was submitted twice (double payment risk)
    #
    # Financial impact is half of the total duplicated amount — representing the
    # value of the potential duplicate payment (one of the two invoices should
    # not be paid).
    #
    # SQL logic:
    #   - Group by vendor_id + invoice_number
    #   - HAVING count > 1 means this (vendor, invoice_number) pair appears more than once
    #   - ORDER BY count DESC to surface the most-duplicated invoices first
    rows = cur.execute("""
        SELECT vendor_id, invoice_number, COUNT(*) AS count,
               SUM(amount) AS total_amount
        FROM invoices
        GROUP BY vendor_id, invoice_number
        HAVING count > 1
        ORDER BY count DESC
    """).fetchall()

    if rows:
        a = Anomaly(
            rule_id="DUP-001",
            severity="high",
            category="duplicate",
            description="Duplicate invoice numbers from the same vendor — potential double payment risk",
            affected_records=[dict(r) for r in rows[:20]],
            count=len(rows),
            # Divide by 2: one of the two invoices is legitimate, one is the duplicate.
            # The financial impact is the amount that would be incorrectly paid.
            estimated_financial_impact=sum(r["total_amount"] for r in rows) / 2,
            remediation="Block payment on duplicates; confirm with vendor which is the valid invoice"
        )
        anomalies.append(a)

    # ── RULE 6: AP-002 — Invoices approved without 3-way match ───────────────
    # Severity: high
    # Category: 3way_match
    #
    # A 3-way match requires three documents to agree before an invoice is paid:
    #   1. Purchase Order  — authorises the spend
    #   2. Goods Receipt   — confirms the goods/services were received
    #   3. Invoice         — requests payment from the vendor
    #
    # If an invoice is approved but there is no goods receipt for its PO,
    # the company is potentially paying for goods it hasn't confirmed receiving.
    # This is a common control bypass in AP fraud scenarios.
    #
    # SQL logic:
    #   - Select approved invoices
    #   - LEFT JOIN goods_receipts on the invoice's po_id
    #   - WHERE gr.id IS NULL means no receipt exists for that PO
    #   - ORDER BY amount DESC to surface the highest-value unapproved invoices first
    rows = cur.execute("""
        SELECT i.id AS invoice_id, i.invoice_number, i.amount,
               v.name AS vendor_name, po.status AS po_status
        FROM invoices i
        JOIN vendors v   ON v.id  = i.vendor_id
        JOIN purchase_orders po ON po.id = i.po_id
        LEFT JOIN goods_receipts gr ON gr.po_id = i.po_id
        WHERE i.status = 'approved'
          AND gr.id IS NULL
        ORDER BY i.amount DESC
    """).fetchall()

    if rows:
        a = Anomaly(
            rule_id="AP-002",
            severity="high",
            category="3way_match",
            description="Invoices approved with no goods receipt — 3-way match not completed",
            affected_records=[dict(r) for r in rows[:20]],
            count=len(rows),
            # Total value of approved invoices without receipt confirmation
            estimated_financial_impact=sum(r["amount"] for r in rows),
            remediation="Require goods receipt confirmation before re-approving; consider hold"
        )
        anomalies.append(a)

    # ── RULE 7: AP-003 — Pending invoices past their due date ─────────────────
    # Severity: medium
    # Category: AP_control
    #
    # An invoice that is still 'pending' (not yet approved or rejected) after
    # its due date indicates a workflow stall — the approval process has not
    # completed in time. This can result in:
    #   - Late payment penalties charged by the vendor
    #   - Damage to vendor relationships
    #   - Inaccurate AP aging reports
    #
    # We use today's ISO date as a parameterised query value (?), which is
    # safer than string interpolation and avoids SQL injection.
    #
    # SQL logic:
    #   - Select invoices with status 'pending'
    #   - WHERE due_date < today (date stored as ISO string, so string comparison works)
    #   - ORDER BY due_date to surface the most overdue invoices first
    today = datetime.date.today().isoformat()  # e.g. "2026-03-31"

    rows = cur.execute("""
        SELECT i.id, i.invoice_number, i.amount, i.due_date,
               v.name AS vendor_name
        FROM invoices i
        JOIN vendors v ON v.id = i.vendor_id
        WHERE i.status = 'pending'
          AND i.due_date < ?
        ORDER BY i.due_date
    """, (today,)).fetchall()
    # (today,) is a single-element tuple — the parameterised value for the ? placeholder

    if rows:
        a = Anomaly(
            rule_id="AP-003",
            severity="medium",
            category="AP_control",
            description="Invoices past due date still in 'pending' status (workflow stall)",
            affected_records=[dict(r) for r in rows[:20]],
            count=len(rows),
            # Total value of stalled invoices — exposure if vendors charge late fees
            estimated_financial_impact=sum(r["amount"] for r in rows),
            remediation="Escalate to AP manager; late payment may trigger vendor penalties"
        )
        anomalies.append(a)

    # ── RULE 8: VND-001 — Open invoices from inactive vendors ─────────────────
    # Severity: medium
    # Category: AP_control
    #
    # If a vendor has been deactivated (is_active = 0), they should not have
    # any open (unpaid, uncancelled) invoices. Open invoices from inactive vendors
    # risk paying a decommissioned supplier — potentially fraudulently reactivated
    # or left over from an incomplete offboarding process.
    #
    # SQL logic:
    #   - Join invoices to vendors
    #   - WHERE is_active = 0 (inactive vendor)
    #   - AND status NOT IN ('paid', 'cancelled') — still open/actionable
    rows = cur.execute("""
        SELECT i.id, i.invoice_number, i.amount, i.status,
               v.name AS vendor_name
        FROM invoices i
        JOIN vendors v ON v.id = i.vendor_id
        WHERE v.is_active = 0
          AND i.status NOT IN ('paid', 'cancelled')
    """).fetchall()

    if rows:
        a = Anomaly(
            rule_id="VND-001",
            severity="medium",
            category="AP_control",
            description="Open invoices from inactive vendors — risk of paying decommissioned suppliers",
            affected_records=[dict(r) for r in rows[:20]],
            count=len(rows),
            # Total value at risk of being paid to an inactive/decommissioned vendor
            estimated_financial_impact=sum(r["amount"] for r in rows),
            remediation="Confirm vendor deactivation reason; cancel or re-route open invoices"
        )
        anomalies.append(a)

   
    # Close the database connection and release the file lock
    con.close()

    # Return the full list of Anomaly instances found across all 8 rules
    return anomalies


# ── build_report ──────────────────────────────────────────────────────────────

from anomaly_agent_helpers.build_report import build_report

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    """
    CLI entry point for anomaly_agent.py.

    Orchestrates the two-step pipeline:
        1. run_checks()   — execute all 8 SQL anomaly rules against the database
        2. build_report() — assemble the results into a structured JSON report

    Then writes the report to disk and logs a summary of findings.

    CLI arguments:
        --db   Path to the SQLite database (default: p2p.db)
        --out  Path for the output JSON report (default: anomaly_report.json)
    """

    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",  default="p2p.db")           # database path
    parser.add_argument("--out", default="anomaly_report.json")  # output report path

    # Parse the CLI arguments from sys.argv
    args = parser.parse_args()

    # Print to console (not log) so this appears even if log level is high
    print("🔎 Running data integrity checks…")

    # Run all 8 anomaly detection rules against the database
    anomalies = run_checks(args.db)

    # Assemble the final report dict from the anomaly list
    report = build_report(anomalies)

    # Write the report to disk as formatted JSON (indent=2 for readability)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary header to console
    print(f"\n📋 Anomaly Report Summary:")

    # Shorthand reference to the summary section of the report
    s = report["summary"]

    # Log summary metrics at INFO level — appears on console and in log file
    log.info(f"   Total anomalies: {s['total_anomalies_found']}")
    log.info(f"   Critical: {s['critical_count']}  High: {s['high_count']}")

    # Format the financial exposure with commas and 2 decimal places
    log.info(f"   Estimated financial exposure: ${s['total_estimated_financial_exposure']:,.2f}")
    log.info(f"\n✅ Full report written → {args.out}")

    # Log each individual anomaly with a coloured severity icon
    for a in anomalies:
        # Select icon based on severity:
        #   🔴 = critical (GL missing, GL imbalanced)
        #   🟠 = high (over-received, credit limit, duplicate, 3-way match)
        #   🟡 = medium (workflow stall, inactive vendor) — default for anything else
        icon = "🔴" if a.severity == "critical" else "🟠" if a.severity == "high" else "🟡"

        log.info(f"  {icon} [{a.rule_id}] {a.description} ({a.count} records)")


# ── Entry point ───────────────────────────────────────────────────────────────

# Only call main() when this script is run directly (python anomaly_agent.py).
# When anomaly_agent is imported by another module (e.g. in tests or pipelines),
# main() is not called automatically.
if __name__ == "__main__":
    main()
