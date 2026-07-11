"""
context_packer.py — Stretch goal: given an invoice ID, assemble the richest
possible context about that invoice that fits in a target token budget.

Purpose:
    LLMs have a finite context window. When an AI agent needs to reason about
    a specific invoice, we can't dump the entire database into the prompt.
    This script solves that by assembling the most relevant information about
    one invoice into a single LLM-ready string, respecting a token budget.

    It uses a priority-ranked greedy packing strategy:
        Priority 1 — Invoice core facts        (always included, never dropped)
        Priority 2 — PO, goods receipts, GL    (always included, truncated if needed)
        Priority 3 — Known anomalies           (included if space allows)
        Priority 4 — Vendor invoice history    (included if space allows, dropped first)

    This ensures the most financially critical information is always present,
    while lower-signal context is dropped gracefully when the budget runs out.

Usage:
    python context_packer.py --db p2p.db --invoice-id 42 --token-budget 4000
    python context_packer.py --db p2p.db --invoice-id 42 --anomalies anomaly_report.json
"""


from logger import get_logger
log = get_logger("context_packer")


import sqlite3, json, argparse, textwrap

# Conservative estimate of how many characters map to one LLM token.
# Real tokenisers (tiktoken, etc.) give exact counts, but 4 chars/token is a
# reliable rule-of-thumb for English prose with numbers and punctuation.
# Using 4 (not 3 or 5) errs on the conservative side — we'd rather pack
# slightly less than blow the token budget.
APPROX_CHARS_PER_TOKEN = 4


# ── pack_context ──────────────────────────────────────────────────────────────

def pack_context(db_path: str, invoice_id: int,
                 anomaly_report_path: str = None,
                 token_budget: int = 4000) -> str:
    """
    Assemble the richest possible context for a specific invoice that fits
    within the given token budget, using a priority-ranked greedy packing strategy.

    Queries six data sources (invoice, PO, receipts, GL, anomalies, vendor history)
    and packs them into a single LLM-ready string in priority order. Lower-priority
    blocks are dropped or truncated if the budget runs out.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (e.g. "p2p.db").
    invoice_id : int
        The internal invoice ID to pack context for.
    anomaly_report_path : str, optional
        Path to anomaly_report.json. If provided, any anomalies touching this
        invoice are included in the context. If None or missing, skipped silently.
    token_budget : int
        Maximum number of tokens the assembled context may use (default 4000).
        4000 tokens leaves headroom for a system prompt and the LLM's response
        in a typical 8K context window.

    Returns
    -------
    str
        A formatted, LLM-ready string containing all assembled context blocks,
        ending with a token count summary line.
        Returns an error string if the invoice_id is not found.

    Packing strategy:
        Blocks are sorted by priority (1 = highest) and packed greedily.
        Priority 1-2 blocks are always included; if they would exceed the budget
        they are truncated rather than dropped.
        Priority 3-4 blocks are dropped entirely if they don't fit.
    """

    # Convert the token budget to a character budget using our approximation.
    # All length checks in this function use char_budget, not token_budget,
    # to avoid computing tokens on every string operation.
    char_budget = token_budget * APPROX_CHARS_PER_TOKEN

    # Open the SQLite database connection
    con = sqlite3.connect(db_path)

    # sqlite3.Row makes columns accessible by name (r["amount"]) rather than
    # index (r[0]) — essential for building the formatted output strings
    con.row_factory = sqlite3.Row

    # Create a cursor for executing SQL queries
    cur = con.cursor()

    # blocks accumulates (priority, text) tuples as we query each data source.
    # Priority is an integer: lower = more important = packed first.
    # Type hint: list of 2-tuples where first element is int, second is str.
    blocks: list[tuple[int, str]] = []

    # ── Block 1: Invoice core (priority 1 — always include) ──────────────────
    # This is the minimum viable context — invoice number, amount, status,
    # and the vendor's key attributes. Always included, never dropped.
    #
    # SQL: join invoices to vendors so we get vendor details in one query
    # rather than making two separate round-trips to the database.
    # The ? placeholder is parameterised — prevents SQL injection and lets
    # SQLite optimise the query plan.
    inv = cur.execute("""
        SELECT i.*, v.name AS vendor_name, v.payment_terms, v.credit_limit,
               v.category AS vendor_category, v.is_active
        FROM invoices i JOIN vendors v ON v.id = i.vendor_id
        WHERE i.id = ?
    """, (invoice_id,)).fetchone()
    # fetchone() returns a single sqlite3.Row or None if no row matched

    # If no invoice was found, return an error string immediately.
    # The caller (main or an AI agent) can detect this by checking for "ERROR:".
    if not inv:
        return f"ERROR: Invoice ID {invoice_id} not found."

    # Convert the sqlite3.Row to a plain dict so we can use standard dict access.
    # sqlite3.Row supports both column-name and index access, but dict is more
    # explicit and easier to debug.
    inv = dict(inv)

    # Build the invoice core block as a formatted multiline string.
    # textwrap.dedent() removes the leading spaces caused by the indentation
    # of the f-string inside the function body.
    # .strip() removes the leading and trailing newlines added by the triple-quote.
    block1 = textwrap.dedent(f"""
    === INVOICE CORE ===
    Invoice Number : {inv['invoice_number']}
    Invoice ID     : {inv['id']}
    Amount         : ${inv['amount']:,.2f}
    Status         : {inv['invoice_status'] if 'invoice_status' in inv else inv['status']}
    Due Date       : {inv['due_date']}
    Vendor         : {inv['vendor_name']} (ID {inv['vendor_id']})
    Payment Terms  : {inv['payment_terms']}
    Vendor Active  : {'Yes' if inv['is_active'] else 'NO — inactive vendor'}
    Credit Limit   : ${inv['credit_limit']:,.2f}
    """).strip()
    # Note: 'invoice_status' vs 'status' — the SELECT * from invoices returns
    # 'status', but if the query were aliased it might be 'invoice_status'.
    # The conditional handles both column name variants defensively.

    # Add block1 with priority 1 — the highest priority, always packed first
    blocks.append((1, block1))

    # ── Block 2: PO + line items (priority 2) ────────────────────────────────
    # The purchase order that this invoice is billed against, plus its line items.
    # Critical for understanding what was authorised and whether the invoice
    # amount matches the PO value.
    #
    # SQL: simple lookup by PO ID from the invoice record
    po = cur.execute("""
        SELECT * FROM purchase_orders WHERE id = ?
    """, (inv["po_id"],)).fetchone()
    # inv["po_id"] — the PO that this invoice references

    # Query line items for this PO, including a computed line_total and
    # an over-receipt flag for easy anomaly spotting in the context.
    # CASE WHEN ... END produces 'OVER-RECEIVED' or 'ok' as a string column.
    lines = cur.execute("""
        SELECT sku, description, qty_ordered, qty_received, unit_cost,
               (qty_ordered * unit_cost) AS line_total,
               CASE WHEN qty_received > qty_ordered THEN 'OVER-RECEIVED' ELSE 'ok' END AS receipt_flag
        FROM po_line_items WHERE po_id = ?
    """, (inv["po_id"],)).fetchall()
    # fetchall() returns all matching rows as a list of sqlite3.Row objects

    # Only build the PO block if the PO was found (it should always exist
    # if the DB is well-formed, but defensive programming prevents a crash
    # if the FK relationship is broken — common in legacy schemas).
    if po:
        # Convert the PO row to a plain dict for f-string access
        po = dict(po)

        # Build a formatted line per SKU, joining qty/cost/flag info
        lines_text = "\n".join([
            f"  {l['sku']}: {l['qty_ordered']} ordered / {l['qty_received']} received "
            f"@ ${l['unit_cost']:.2f} = ${l['line_total']:,.2f}  [{l['receipt_flag']}]"
            for l in lines
        ])
        # list comprehension iterates over all line item rows and formats each one
        # {l['line_total']:,.2f} — comma-separated thousands, 2 decimal places

        # Build the PO block string using dedent + strip (same pattern as block1)
        block2 = textwrap.dedent(f"""
        === PURCHASE ORDER ===
        PO ID    : {po['id']}
        Status   : {po['status']}
        Created  : {po['created_at']} by {po['created_by']}
        Notes    : {po['notes'] or 'none'}
        Line Items:
        {lines_text}
        """).strip()
        # po['notes'] or 'none' — if notes is None, display 'none' instead of 'None'

        # Add with priority 2 — high priority, always included if budget allows
        blocks.append((2, block2))

    # ── Block 3: Goods receipts (priority 2) ─────────────────────────────────
    # Shows whether goods were physically received for this invoice's PO.
    # Essential for 3-way match validation — if no receipt exists, the match
    # is incomplete and the invoice should not have been approved.
    #
    # SQL: LEFT JOIN receipt_lines to count how many line items are on each receipt.
    # GROUP BY gr.id aggregates the count per receipt header.
    # COUNT(rl.id) counts receipt line rows — returns 0 if no lines (LEFT JOIN).
    receipts = cur.execute("""
        SELECT gr.id, gr.received_by, gr.received_at, COUNT(rl.id) AS line_count
        FROM goods_receipts gr
        LEFT JOIN receipt_lines rl ON rl.receipt_id = gr.id
        WHERE gr.po_id = ?
        GROUP BY gr.id
    """, (inv["po_id"],)).fetchall()

    if receipts:
        # At least one receipt exists — build a summary line per receipt
        rec_text = "\n".join([
            f"  Receipt {r['id']}: received {r['received_at']} by {r['received_by']} ({r['line_count']} lines)"
            for r in receipts
        ])
        # Mark the 3-way match as COMPLETE since a receipt exists
        block3 = f"=== GOODS RECEIPTS ===\n{rec_text}\n3-Way Match: COMPLETE"
    else:
        # No receipt found — explicitly flag the 3-way match as incomplete.
        # This is critical context for an AI agent reasoning about the invoice.
        block3 = "=== GOODS RECEIPTS ===\nNO RECEIPT ON FILE — 3-way match INCOMPLETE"

    # Add with priority 2 regardless of whether receipts exist —
    # the absence of a receipt is just as important to include as the presence
    blocks.append((2, block3))

    # ── Block 4: GL entries (priority 2) ─────────────────────────────────────
    # Shows the accounting entries posted for this invoice.
    # Includes a balance check — if debits ≠ credits the GL is imbalanced,
    # which is a GL-001 anomaly and a critical accounting error.
    #
    # SQL: LEFT JOIN account_codes to get the human-readable account name
    # alongside the raw account code. LEFT JOIN means rows without a matching
    # account code still appear (defensive against missing reference data).
    gl_rows = cur.execute("""
        SELECT g.account_code, ac.name AS account_name, g.debit, g.credit, g.posted_at
        FROM gl_entries g
        LEFT JOIN account_codes ac ON ac.code = g.account_code
        WHERE g.invoice_id = ?
    """, (invoice_id,)).fetchall()

    if gl_rows:
        # Sum debit and credit totals across all GL rows for this invoice
        total_d = sum(r["debit"] for r in gl_rows)
        total_c = sum(r["credit"] for r in gl_rows)

        # Check balance: ABS difference < 0.01 (one cent) accounts for floating-point
        # rounding in financial calculations. Anything larger is a real imbalance.
        # The Δ prefix (delta) is a conventional accounting symbol for "difference".
        balanced = "BALANCED" if abs(total_d - total_c) < 0.01 else f"IMBALANCED (Δ${abs(total_d-total_c):.2f})"

        # Build one line per GL entry showing account code, name, debit, credit, date
        gl_text = "\n".join([
            f"  {r['account_code']} {r['account_name']}: Dr ${r['debit']:.2f} / Cr ${r['credit']:.2f}  [{r['posted_at']}]"
            for r in gl_rows
        ])
        # Dr = Debit, Cr = Credit — standard accounting abbreviations

        # Combine the GL lines with the balance summary
        block4 = f"=== GL ENTRIES ===\n{gl_text}\nTotal: Dr ${total_d:.2f} / Cr ${total_c:.2f} — {balanced}"
    else:
        # No GL entries posted — this is an AP-001 anomaly for approved/paid invoices
        block4 = "=== GL ENTRIES ===\nNO GL ENTRIES POSTED"

    # Add with priority 2 — GL absence is important context, always include
    blocks.append((2, block4))

    # ── Block 5: Anomalies touching this invoice (priority 3) ─────────────────
    # Cross-references the anomaly report to find any rules that flagged this
    # specific invoice. This gives the AI agent explicit anomaly context without
    # having to re-derive it from the raw data.
    #
    # This block is optional — if no anomaly report path was provided, or the
    # file can't be read, we skip silently rather than crashing.
    if anomaly_report_path:
        try:
            # Load the full anomaly report JSON from disk
            with open(anomaly_report_path) as f:
                report = json.load(f)

            # hits accumulates formatted strings for each anomaly that touched this invoice
            hits = []

            # Iterate over every anomaly in the report
            for a in report.get("anomalies", []):
                # Each anomaly has an affected_records list — check if this invoice
                # appears in it. We check both "invoice_id" and "id" because different
                # rules use different column names in their affected_records.
                for rec in a.get("affected_records", []):
                    if rec.get("invoice_id") == invoice_id or rec.get("id") == invoice_id:
                        # This anomaly flagged our invoice — add a summary line
                        hits.append(f"  [{a['rule_id']}] {a['severity'].upper()}: {a['description']}")
                        # break out of the inner loop — we only need one match per anomaly
                        break

            # Only add the anomaly block if at least one anomaly was found
            if hits:
                # Priority 3 — lower than core data but higher than vendor history
                blocks.append((3, "=== ANOMALIES ===\n" + "\n".join(hits)))

        except Exception:
            # Silently skip if the file doesn't exist, is malformed, or any other error.
            # We use bare Exception (not FileNotFoundError) because any failure here
            # is non-fatal — we simply omit the anomaly block from the context.
            pass

    # ── Block 6: Vendor invoice history (priority 4 — lowest priority) ────────
    # The last 10 invoices from this vendor (excluding the current one).
    # Gives the AI agent pattern recognition context — are there repeated disputes?
    # Is this vendor consistently overdue? Are amounts in a typical range?
    #
    # SQL: exclude the current invoice (id != ?) to avoid showing it twice.
    # ORDER BY due_date DESC shows the most recent invoices first.
    # LIMIT 10 keeps the block short enough to fit within the token budget.
    history = cur.execute("""
        SELECT invoice_number, amount, status, due_date
        FROM invoices
        WHERE vendor_id = ? AND id != ?
        ORDER BY due_date DESC
        LIMIT 10
    """, (inv["vendor_id"], invoice_id)).fetchall()
    # Two parameterised values: vendor_id to filter by vendor, invoice_id to exclude self

    if history:
        # Build one formatted line per historical invoice
        hist_text = "\n".join([
            f"  {r['invoice_number']}: ${r['amount']:,.2f}  [{r['status']}]  due {r['due_date']}"
            for r in history
        ])
        # Priority 4 — lowest priority, dropped first if the budget is tight
        blocks.append((4, f"=== VENDOR INVOICE HISTORY (last 10) ===\n{hist_text}"))

    # Close the database connection — releases the file lock so other processes
    # can access the DB. Always close when done to avoid "database is locked" errors.
    con.close()

    # ── Greedy packing by priority ─────────────────────────────────────────────
    # Now that we have all blocks, pack them into the final output string
    # in priority order, stopping when the character budget is exhausted.

    # List of text blocks that will be joined into the final output
    output_parts = []

    # Track how many characters we've used so far
    used_chars = 0

    # Build the header line that identifies which invoice this context is for
    header = f"# P2P Context: Invoice {inv['invoice_number']} (ID {invoice_id})\n\n"

    # Deduct the header length from the budget upfront
    used_chars += len(header)

    # Sort blocks by priority (ascending — lower number = higher priority = packed first)
    # lambda x: x[0] extracts the priority integer from each (priority, text) tuple
    for priority, text in sorted(blocks, key=lambda x: x[0]):
        # Calculate how many chars this block needs, +2 for the "\n\n" separator
        needed = len(text) + 2

        if used_chars + needed <= char_budget:
            # Block fits within the remaining budget — include it in full
            output_parts.append(text)
            used_chars += needed

        elif priority <= 2:
            # Block is high priority (1 or 2) but doesn't fit in full.
            # Truncate it to fit rather than dropping it entirely.
            # -50 leaves a small buffer for the truncation notice string.
            remaining = char_budget - used_chars - 50

            if remaining > 100:
                # Only truncate if there's at least 100 chars of meaningful content.
                # Less than 100 chars would be too short to be useful.
                output_parts.append(text[:remaining] + "\n[… truncated to fit token budget]")
                # Mark the budget as exhausted so no more blocks are attempted
                used_chars = char_budget

        # Priority 3+ blocks that don't fit are silently dropped.
        # This is intentional — anomaly flags and vendor history are "nice to have"
        # but not essential for the AI agent's core reasoning about the invoice.

    # Join all packed blocks with double newlines as separators
    final = header + "\n\n".join(output_parts)

    # Calculate the approximate token count of the final assembled string
    # Integer division (//) gives a whole number without a decimal
    approx_tokens = len(final) // APPROX_CHARS_PER_TOKEN

    # Append a footer showing how much of the budget was used.
    # This helps the AI agent and developers understand the context coverage.
    final += f"\n\n---\n[Context assembled: ~{approx_tokens} tokens / {token_budget} budget]"

    return final


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    """
    CLI entry point for context_packer.py.

    Parses CLI arguments, calls pack_context() to assemble the invoice context,
    logs the result, then calls the Claude invoice summary helper to produce
    an AI analysis and saves it to a text file.

    CLI arguments:
        --db            Path to the SQLite database (default: p2p.db)
        --invoice-id    Invoice ID to pack context for (required)
        --anomalies     Path to anomaly_report.json (default: anomaly_report.json)
        --token-budget  Maximum tokens for the assembled context (default: 4000)
    """

    parser = argparse.ArgumentParser()

    # Database path — defaults to p2p.db in the current directory
    parser.add_argument("--db", default="p2p.db")

    # Invoice ID — required, must be an integer.
    # type=int tells argparse to convert the string argument to int automatically.
    parser.add_argument("--invoice-id", type=int, required=True)

    # Path to the anomaly report — used to cross-reference anomalies for this invoice
    parser.add_argument("--anomalies", default="anomaly_report.json")

    # Token budget — controls how much context is packed into the output string
    parser.add_argument("--token-budget", type=int, default=4000)

    # Parse the arguments from sys.argv into an args namespace object
    args = parser.parse_args()

    # Call pack_context() with all four arguments.
    # The result is a formatted string ready to be passed directly to an LLM.
    ctx = pack_context(
        args.db,
        args.invoice_id,
        anomaly_report_path=args.anomalies,
        token_budget=args.token_budget
    )

    # Log the full assembled context at INFO level — appears on console and in log file
    log.info(ctx)

    # Import the Claude invoice summary helper from the helpers subpackage.
    # The import is inside main() so it's only loaded when this script runs directly,
    # not when pack_context() is imported by other scripts.
    # Note: 'claue_invoice_summary' appears to be a typo for 'claude_invoice_summary'
    # in the original code — kept as-is to match the actual helper module.
    from context_packer_helpers.invoice_summary import claue_invoice_summary

    # Call Claude to produce a structured analysis of the packed context.
    # This sends the assembled context string to the API and returns an analysis.
    claude_analysis = claue_invoice_summary(ctx)

    # Write the Claude analysis to a plain text file for review or downstream use
    with open("invoice_summary_ai_analysis.txt", 'w') as f:
        f.write(claude_analysis)

    # Log the Claude analysis at INFO level — appears on console and in log file
    log.info(claude_analysis)


# ── Entry point ───────────────────────────────────────────────────────────────

# Only call main() when this script is run directly (python context_packer.py).
# When pack_context() is imported by other scripts (e.g. app.py or tests),
# main() is not called automatically — only the function is available.
if __name__ == "__main__":
    main()
