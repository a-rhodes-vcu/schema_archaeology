"""
knowledge_graph.py — Map Vendor → PO → Invoice → GLEntry relationships
using networkx and identify the vendor with the deepest open transaction chain.
"""

import sqlite3
import networkx as nx
from collections import defaultdict

def build_graph(db_path: str = "p2p.db") -> nx.DiGraph:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    G = nx.DiGraph()

    # Vendors
    for r in cur.execute("SELECT id, name FROM vendors").fetchall():
        G.add_node(f"V:{r['id']}", label=r['name'], type="vendor")

    # POs — edge Vendor → PO
    for r in cur.execute(
        "SELECT id, vendor_id, status FROM purchase_orders WHERE status NOT IN ('cancelled')"
    ).fetchall():
        G.add_node(f"PO:{r['id']}", type="po", status=r['status'])
        G.add_edge(f"V:{r['vendor_id']}", f"PO:{r['id']}")

    # Invoices — edge PO → Invoice (open only)
    for r in cur.execute(
        "SELECT id, po_id, vendor_id, status FROM invoices WHERE status NOT IN ('paid','cancelled')"
    ).fetchall():
        G.add_node(f"INV:{r['id']}", type="invoice", status=r['status'])
        G.add_edge(f"PO:{r['po_id']}", f"INV:{r['id']}")

    # GL entries — edge Invoice → GLEntry
    for r in cur.execute("SELECT id, invoice_id FROM gl_entries").fetchall():
        G.add_node(f"GL:{r['id']}", type="gl")
        G.add_edge(f"INV:{r['invoice_id']}", f"GL:{r['id']}")

    con.close()
    return G


def deepest_vendor_chain(G: nx.DiGraph) -> dict:
    """
    For each vendor node, find the longest path to any GL entry
    in the subgraph of open transactions. Returns the top 10.
    """
    results = []

    vendor_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "vendor"]

    for vendor in vendor_nodes:
        # Get all descendants reachable from this vendor
        descendants = nx.descendants(G, vendor)
        if not descendants:
            continue

        # Find the longest path from vendor to any descendant
        max_depth = 0
        for target in descendants:
            try:
                path_len = nx.shortest_path_length(G, vendor, target)
                max_depth = max(max_depth, path_len)
            except nx.NetworkXNoPath:
                continue

        results.append({
            "vendor":         G.nodes[vendor]["label"],
            "vendor_id":      vendor,
            "max_depth":      max_depth,
            "total_nodes":    len(descendants),
        })

    return sorted(results, key=lambda x: x["max_depth"], reverse=True)[:10]


if __name__ == "__main__":
    print("Building knowledge graph…")
    G = build_graph()
    print(f"Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

    print("\nTop 10 vendors by dependency chain depth:")
    for r in deepest_vendor_chain(G):
        print(f"  depth={r['max_depth']}  nodes={r['total_nodes']:>4}  {r['vendor']}")