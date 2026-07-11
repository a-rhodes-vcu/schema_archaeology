"""
seed_db.py — Generate a realistic SQLite P2P database with ~500 vendors,
~2000 POs, ~5000 invoices and intentional financial anomalies.
Run once before the interview: python seed_db.py
"""

import sqlite3, random, string, datetime, os

DB_PATH = "p2p.db"

CATEGORIES = ["Software", "Hardware", "Consulting", "Logistics", "Office Supplies", "Facilities"]
PAYMENT_TERMS = ["NET30", "NET60", "NET90", "NET15", "COD"]
STATUSES_PO = ["draft", "approved", "received", "closed", "cancelled"]
STATUSES_INV = ["pending", "approved", "paid", "disputed", "overdue"]
# 5 account codes seeded
ACCOUNT_CODES = [
    ("2000", "AP Control",    "liability"),
    ("5000", "COGS",          "expense"),
    ("6000", "Opex Expense",  "expense"),
    ("6100", "SaaS Expense",  "expense"),
    ("1200", "Prepaid Assets","asset"),
]

def rand_date(start_days_ago=365, end_days_ago=0):
    delta = random.randint(end_days_ago, start_days_ago)
    return (datetime.date.today() - datetime.timedelta(days=delta)).isoformat()

def rand_invoice_number():
    return "INV-" + "".join(random.choices(string.digits, k=4))


def build():
    """
    Deletes the physical p2p.db, but only if it already exists.
    Creates a brand new empty database file
    This approach has one important implication — every run of seed_db.py wipes all existing data. 
    Since the data is randomly generated, you'll get a different set of vendors, invoices, and anomalies each time.
    NOTE: If we want a reproducible dataset across runs, add a fixed random seed at the top of build()
    """
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
    CREATE TABLE vendors (
        id INTEGER PRIMARY KEY, name TEXT, payment_terms TEXT,
        credit_limit REAL, category TEXT, is_active INTEGER
    );
    CREATE TABLE purchase_orders (
        id INTEGER PRIMARY KEY, vendor_id INTEGER, status TEXT,
        created_by TEXT, created_at TEXT, notes TEXT
    );
    CREATE TABLE po_line_items (
        id INTEGER PRIMARY KEY, po_id INTEGER, sku TEXT, description TEXT,
        qty_ordered REAL, qty_received REAL, unit_cost REAL
    );
    CREATE TABLE goods_receipts (
        id INTEGER PRIMARY KEY, po_id INTEGER, received_by TEXT, received_at TEXT
    );
    CREATE TABLE receipt_lines (
        id INTEGER PRIMARY KEY, receipt_id INTEGER, sku TEXT, qty_received REAL
    );
    CREATE TABLE invoices (
        id INTEGER PRIMARY KEY, vendor_id INTEGER, po_id INTEGER,
        invoice_number TEXT, amount REAL, status TEXT, due_date TEXT
    );
    CREATE TABLE gl_entries (
        id INTEGER PRIMARY KEY, invoice_id INTEGER, account_code TEXT,
        debit REAL, credit REAL, posted_at TEXT
    );
    CREATE TABLE account_codes (
        id INTEGER PRIMARY KEY, code TEXT, name TEXT, type TEXT
    );
    """)

    # Account codes
    cur.executemany("INSERT INTO account_codes(code,name,type) VALUES(?,?,?)", ACCOUNT_CODES)

    # Vendors (500)
    vendors = []
    for i in range(1, 501):
        name = f"Vendor {i} {random.choice(['Corp','LLC','Inc','Ltd','Partners'])}"
        terms = random.choice(PAYMENT_TERMS)
        limit = round(random.choice([50_000, 100_000, 250_000, 500_000, 1_000_000]), 2)
        cat = random.choice(CATEGORIES)
        active = 1 if random.random() > 0.05 else 0
        vendors.append((i, name, terms, limit, cat, active))
    cur.executemany("INSERT INTO vendors VALUES(?,?,?,?,?,?)", vendors)

    # POs (2000), line items, goods receipts, receipt lines
    po_id = 1
    li_id = 1
    gr_id = 1
    rl_id = 1
    po_rows, li_rows, gr_rows, rl_rows = [], [], [], []
    users = [f"user{i}@company.com" for i in range(1, 21)]

    for _ in range(2000):
        vendor_id = random.randint(1, 500)
        status = random.choices(STATUSES_PO, weights=[5, 30, 40, 20, 5])[0]
        created_at = rand_date(400, 1)
        po_rows.append((po_id, vendor_id, status, random.choice(users), created_at, None))

        skus = [f"SKU-{random.randint(1000,9999)}" for _ in range(random.randint(1,5))]
        for sku in skus:
            qty_ord = random.randint(1, 100)
            # Anomaly: ~3% of lines have received > ordered
            qty_rec = qty_ord if random.random() > 0.03 else qty_ord + random.randint(1, 10)
            cost = round(random.uniform(10, 5000), 2)
            li_rows.append((li_id, po_id, sku, f"Item {sku}", qty_ord, qty_rec, cost))
            li_id += 1

        if status in ("received", "closed"):
            received_at = rand_date(200, 0)
            gr_rows.append((gr_id, po_id, random.choice(users), received_at))
            for sku in skus:
                qty = random.randint(1, 50)
                rl_rows.append((rl_id, gr_id, sku, qty))
                rl_id += 1
            gr_id += 1
        po_id += 1

    cur.executemany("INSERT INTO purchase_orders VALUES(?,?,?,?,?,?)", po_rows)
    cur.executemany("INSERT INTO po_line_items VALUES(?,?,?,?,?,?,?)", li_rows)
    cur.executemany("INSERT INTO goods_receipts VALUES(?,?,?,?)", gr_rows)
    cur.executemany("INSERT INTO receipt_lines VALUES(?,?,?,?)", rl_rows)

    # Invoices (5000) and GL entries
    inv_rows, gl_rows = [], []
    inv_id = 1
    gl_id = 1
    invoice_numbers_by_vendor: dict[int, list[str]] = {}

    for _ in range(5000):
        vendor_id = random.randint(1, 500)
        po_id_ref = random.randint(1, 2000)
        # Anomaly: ~2% duplicate invoice numbers per vendor
        if vendor_id in invoice_numbers_by_vendor and random.random() < 0.02:
            inv_num = random.choice(invoice_numbers_by_vendor[vendor_id])
        else:
            inv_num = rand_invoice_number()
            invoice_numbers_by_vendor.setdefault(vendor_id, []).append(inv_num)

        amount = round(random.uniform(500, 50_000), 2)
        status = random.choices(STATUSES_INV, weights=[10, 40, 35, 10, 5])[0]
        due_date = rand_date(180, -60)
        inv_rows.append((inv_id, vendor_id, po_id_ref, inv_num, amount, status, due_date))

        # GL entries: ~8% of approved/paid invoices missing GL (anomaly)
        if status in ("approved", "paid") and random.random() > 0.08:
            ac = random.choice(["5000","6000","6100"])
            gl_rows.append((gl_id, inv_id, "2000",    0,      amount, rand_date(90,0)))
            gl_id += 1
            # Anomaly: ~5% imbalanced GL
            credit_amt = amount if random.random() > 0.05 else amount + random.uniform(1, 500)
            gl_rows.append((gl_id, inv_id, ac, amount, 0, rand_date(90,0)))
            gl_id += 1

        inv_id += 1

    cur.executemany("INSERT INTO invoices VALUES(?,?,?,?,?,?,?)", inv_rows)
    cur.executemany("INSERT INTO gl_entries VALUES(?,?,?,?,?,?)", gl_rows)

    con.commit()
    con.close()
    print(f"✅ Seeded {DB_PATH} — vendors:500  POs:2000  invoices:5000")

if __name__ == "__main__":
    build()
