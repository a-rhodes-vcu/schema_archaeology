import sqlite3

def _detect_implicit_fks(schema: dict) -> list[dict]:
    """Heuristic: col name ends in _id or matches <table>_id pattern → likely FK."""
    fks = []
    table_names = {t for t in schema if not t.startswith("_")}
    for table, meta in schema.items():
        if table.startswith("_"):
            continue
        for col in meta["columns"]:
            cname = col["name"]
            if cname == "id" or col["pk"]:
                continue
            if cname.endswith("_id"):
                ref = cname[:-3]  # strip _id
                if ref in table_names:
                    fks.append({
                        "from_table": table,
                        "from_col": cname,
                        "to_table": ref,
                        "to_col": "id",
                        "confidence": "high",
                        "note": "column name matches <table>_id pattern"
                    })
            # SKU cross-reference special case
            if cname == "sku" and table in ("receipt_lines", "po_line_items"):
                peer = "po_line_items" if table == "receipt_lines" else "receipt_lines"
                if peer in table_names:
                    fks.append({
                        "from_table": table,
                        "from_col": "sku",
                        "to_table": peer,
                        "to_col": "sku",
                        "confidence": "medium",
                        "note": "shared domain key — no FK constraint enforced"
                    })
    # Deduplicate
    seen = set()
    deduped = []
    for fk in fks:
        key = (fk["from_table"], fk["from_col"], fk["to_table"])
        if key not in seen:
            seen.add(key)
            deduped.append(fk)
    return deduped

def crawl_schema(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]

    schema = {}
    for table in tables:
        cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
        columns = [{"name": c[1], "type": c[2], "pk": bool(c[5])} for c in cols]
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        # Increase sample size
        rows = cur.execute(f"SELECT * FROM {table} LIMIT 20").fetchall()
        col_names = [c["name"] for c in columns]
        samples = [dict(zip(col_names, r)) for r in rows]

        # Discover distinct values for low-cardinality columns
        domain_values = {}
        for col in columns:
            distinct_count = cur.execute(
                f"SELECT COUNT(DISTINCT {col['name']}) FROM {table}"
            ).fetchone()[0]
            # If fewer than 20 distinct values, pull them all — likely an enum
            if distinct_count <= 20:
                vals = cur.execute(
                    f"SELECT DISTINCT {col['name']} FROM {table} "
                    f"WHERE {col['name']} IS NOT NULL ORDER BY {col['name']}"
                ).fetchall()
                domain_values[col["name"]] = [r[0] for r in vals]

        # Add row count stats per column (null counts, min, max for numerics)
        col_stats = {}
        for col in columns:
            null_count = cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col['name']} IS NULL"
            ).fetchone()[0]
            stats = {"null_count": null_count, "null_pct": round(null_count / count * 100, 1) if count else 0}
            if col["type"] in ("INTEGER", "REAL"):
                row = cur.execute(
                    f"SELECT MIN({col['name']}), MAX({col['name']}) FROM {table}"
                ).fetchone()
                stats["min"] = row[0]
                stats["max"] = row[1]
            col_stats[col["name"]] = stats

        schema[table] = {
            "columns":      columns,
            "row_count":    count,
            "samples":      samples,
            "domain_values": domain_values,   # ← distinct values for enums
            "col_stats":    col_stats,        # ← null %, min, max
        }

    schema["_implicit_fks"] = _detect_implicit_fks(schema)
    con.close()
    return schema