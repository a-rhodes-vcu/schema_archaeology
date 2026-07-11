import sqlite3

con = sqlite3.connect("p2p.db")

with open("anomaly_checks_per_vendor.sql") as f:
    sql = f.read()

con.executescript(sql)
con.close()