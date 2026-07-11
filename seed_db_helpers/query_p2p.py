import sqlite3

con = sqlite3.connect("p2p.db")
con.row_factory = sqlite3.Row  # lets you access columns by name
cur = con.cursor()

# example: find invoices approved with no GL entry
rows = cur.execute("""
    SELECT i.invoice_number, i.amount, v.name AS vendor
    FROM invoices i
    JOIN vendors v ON v.id = i.vendor_id
    --WHERE i.status = 'approved'
    --  AND i.id NOT IN (SELECT DISTINCT invoice_id FROM gl_entries)
    ORDER BY i.invoice_number, DESC
""").fetchall()
for r in rows:
    print(dict(r))

con.close()