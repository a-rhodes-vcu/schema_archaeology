# P2P Context Engine

A Context Engine for AI agents to reason over Purchase-to-Pay data with financial precision. Built for the Builder Interview Challenge — Data Track.

---

## What this is

A pipeline that takes an undocumented P2P SQLite database and makes it queryable in natural language — with zero hallucinations. When a finance user asks *"Why was invoice INV-2291 flagged?"* or *"Which vendors are creating the most AP risk this quarter?"*, the engine pulls the right context, understands the relationships between entities, and answers with financial precision.

---

## Architecture

```
p2p.db (SQLite)
    │
    ├── seed_db.py          → p2p.db                    seed realistic data + anomalies
    ├── schema_agent.py     → schema_context.json        semantic layer via Claude
    ├── anomaly_agent.py    → anomaly_report.json        8 SQL integrity rules
    ├── rag_pipeline.py     → interactive Q&A            hybrid RAG (SQL + ChromaDB)
    ├── context_packer.py   → LLM-ready string           token-budget context assembly
    ├── knowledge_graph.py  → dependency chain report    networkx vendor graph
    ├── app.py              → Streamlit UI               chat interface
    ├── cache_logger.py     → logs/cache_usage.log       token + cost tracking
    └── logger.py           → logs/<script>.log          shared logging
```

---

## Setup

```bash
# Install dependencies
pip install anthropic chromadb streamlit python-dotenv networkx

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...
# or add to a .env file:
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

---

## Full execution order

```bash
# 1. Seed the database
python seed_db.py

# 2. Build the semantic layer
python schema_agent.py --db p2p.db --out schema_context.json

# 3. Run the anomaly detector
python anomaly_agent.py --db p2p.db --out anomaly_report.json

# 4. Start the RAG Q&A pipeline (CLI)
python rag_pipeline.py --db p2p.db --context schema_context.json

# 5. Or start the Streamlit UI
streamlit run app.py

# 6. Pack context for a specific invoice
python context_packer.py --db p2p.db --invoice-id 42 --anomalies anomaly_report.json

# 7. Build knowledge graph and find deepest vendor chain
python knowledge_graph.py --db p2p.db
```

---

## Scripts

### `seed_db.py` — Generate the database

Creates `p2p.db` with realistic P2P data and intentional financial anomalies. Also serves as the **synthetic data generator** for eval datasets — add `random.seed(42)` for a reproducible dataset.

```bash
python seed_db.py
```

**What it seeds:**

| Table | Rows | Notes |
|---|---|---|
| `vendors` | 500 | Random payment terms, credit limits, categories |
| `purchase_orders` | 2,000 | Weighted status distribution |
| `po_line_items` | ~6,000 | 3% over-received anomaly |
| `goods_receipts` | ~1,183 | Only for received/closed POs |
| `receipt_lines` | ~3,600 | One per SKU per receipt |
| `invoices` | 5,000 | 2% duplicate invoice numbers, 8% missing GL |
| `gl_entries` | ~6,838 | 5% imbalanced debits/credits |
| `account_codes` | 5 | AP Control, COGS, Opex, SaaS, Prepaid |

**Parameterised for eval datasets:**

```python
def build(
    n_vendors:        int   = 500,
    n_pos:            int   = 2000,
    n_invoices:       int   = 5000,
    anomaly_rate_gl:  float = 0.08,
    anomaly_rate_dup: float = 0.02,
    anomaly_rate_qty: float = 0.03,
    seed:             int   = None,
):
    if seed is not None:
        random.seed(seed)
    # ... rest of build() unchanged
```

---

### `schema_agent.py` — Build the semantic layer

Crawls the schema, infers business meaning via Claude, and writes `schema_context.json`. Both the crawl and LLM annotation are cached — only re-runs when forced.

```bash
python schema_agent.py                    # normal run (loads caches)
python schema_agent.py --force            # re-run everything
python schema_agent.py --force-crawl      # re-crawl only
python schema_agent.py --force-llm        # re-annotate only
python schema_agent.py --skip-eval        # skip self-evaluation
```

**Cache files:**

| File | Contents |
|---|---|
| `crawl_schema_cache.json` | Raw schema from SQLite |
| `llm_annotate_cache.json` | Claude's semantic annotation |
| `schema_context.json` | Final output with evaluation scores |

**Three stages:**

1. `crawl_schema()` — extracts columns, types, sample values, domain values, null counts, implicit FKs via `PRAGMA table_info`
2. `llm_annotate()` — sends schema to Claude with a cached system prompt; Claude infers business meaning for every table and column
3. `evaluate_schema_context()` — Claude self-scores the annotation (0–100) and flags accuracy risks

---

### `anomaly_agent.py` — Detect financial anomalies

Runs 8 SQL rules against the database and outputs `anomaly_report.json`.

```bash
python anomaly_agent.py --db p2p.db --out anomaly_report.json
```

**Rules:**

| Rule | Severity | Category | Description |
|---|---|---|---|
| `AP-001` | Critical | GL integrity | Approved invoice with no GL entry |
| `GL-001` | Critical | GL integrity | Debits ≠ credits per invoice |
| `PO-001` | High | 3-way match | Received qty exceeds ordered qty |
| `CR-001` | High | Credit risk | Open invoices exceed vendor credit limit |
| `DUP-001` | High | Duplicate | Same invoice number from same vendor |
| `AP-002` | High | 3-way match | Approved invoice with no goods receipt |
| `AP-003` | Medium | AP control | Pending invoice past due date |
| `VND-001` | Medium | AP control | Open invoice from inactive vendor |

**Output format:**

```json
{
  "report_generated_at": "2026-03-31T14:22:01Z",
  "summary": {
    "total_anomalies_found": 6,
    "critical_count": 2,
    "high_count": 3,
    "total_estimated_financial_exposure": 4821304.22
  },
  "anomalies": [...]
}
```

---

### `rag_pipeline.py` — Hybrid RAG Q&A

Enables natural language querying over invoices using hybrid search (SQL pre-filter + ChromaDB semantic) and Claude synthesis.
Hybrid RAG means combining two fundamentally different retrieval methods in the same pipeline — one structured, one semantic — and using each where it's strongest.

```bash
python rag_pipeline.py --db p2p.db --context schema_context.json
```

**Unit of knowledge:** A joined invoice summary (vendor + PO + receipt status + GL) — not a raw SQL row. Raw rows lose all relationships.

**How a query works:**

1. `_detect_filter()` — extracts structured filters from the question (invoice number, payment terms, status)
2. `sql_filter_invoice_ids()` — runs SQL pre-pass to narrow candidate set
3. `collection.query()` — ChromaDB semantic search over filtered candidates
4. Confidence check — cosine distance > 1.2 = low confidence warning to Claude
5. Claude synthesis — grounded answer using only retrieved context

**ChromaDB persistence:** The index is stored in `./chroma_db/` and survives restarts. First run ~60 seconds; every subsequent run ~1 second.

**Test questions run automatically on startup:**
- *"Which vendors have invoices that were approved without a complete goods receipt?"*
- *"What's the total AP exposure for vendors on NET60 terms?"*
- *"Are there any invoices from the same vendor with duplicate invoice numbers?"*

---

### `context_packer.py` — Token-budget context assembly *(stretch goal)*

Given an invoice ID, assembles the richest possible context that fits in a 4K token budget using a priority-ranked greedy packing strategy. This is the **Context Window Packing** stretch goal.

```bash
python context_packer.py --db p2p.db --invoice-id 42
python context_packer.py --db p2p.db --invoice-id 42 --anomalies anomaly_report.json --token-budget 4000
```

**Priority order (highest signal-to-noise):**

| Priority | Block | If budget runs out |
|---|---|---|
| 1 | Invoice core facts | Truncated, never dropped |
| 2 | PO + line items, goods receipts, GL entries | Truncated, never dropped |
| 3 | Known anomalies from report | Dropped entirely |
| 4 | Vendor invoice history (last 10) | Dropped first |

The output ends with a token count footer showing exactly how much of the budget was used.

---

### `knowledge_graph.py` — Entity dependency graph *(stretch goal)*

Maps `Vendor → PO → Invoice → GLEntry` relationships using `networkx` and identifies which vendor has the deepest dependency chain in current open transactions. This is the **Knowledge Graph** stretch goal.

```bash
pip install networkx
python knowledge_graph.py --db p2p.db
```

**Output:**

```
Building knowledge graph…
Nodes: 14,823  Edges: 18,241

Top 10 vendors by dependency chain depth:
  depth=4  nodes=  87  Vendor 142 Corp
  depth=4  nodes=  63  Vendor 291 LLC
  depth=3  nodes= 124  Vendor 57 Inc
  ...
```

**How it works:**

```python
import networkx as nx

G = nx.DiGraph()

# Vendor → PO
G.add_edge(f"V:{vendor_id}", f"PO:{po_id}")

# PO → Invoice (open only)
G.add_edge(f"PO:{po_id}", f"INV:{invoice_id}")

# Invoice → GL entry
G.add_edge(f"INV:{invoice_id}", f"GL:{gl_id}")

# Find deepest chain per vendor
for vendor in vendor_nodes:
    max_depth = max(
        nx.shortest_path_length(G, vendor, target)
        for target in nx.descendants(G, vendor)
    )
```

---

### `app.py` — Streamlit chat UI

Browser-based chat interface for the RAG pipeline.

```bash
streamlit run app.py
# or with uv:
uv run streamlit run app.py
```

**Features:**
- Chat interface with full message history
- 5 example AP questions in the sidebar
- SQL pre-filter info shown per query
- Confidence signal (OK / Low) per query
- Expandable retrieved chunk inspector with cosine distances
- Response time metric

---

### `cache_logger.py` — Token and cost tracking

Logs Anthropic prompt cache token usage and cost savings to `logs/cache_usage.log`.

```python
from cache_logger import CacheLogger, CacheSession, TokenUsage, timed_call

logger  = CacheLogger("logs/cache_usage.log", model="claude-opus-4-5")
session = CacheSession(logger)

with timed_call() as t:
    response = client.messages.create(...)

usage = TokenUsage.from_response(response.usage, t.elapsed, model="claude-opus-4-5")
session.add(usage, label="llm_annotate")

session.summary()
```

**Log format:**

```
[2026-03-31 14:22:01]  llm_annotate
──────────────────────────────────────────────────────────────
  Cache write tokens   :      8,432
  Cache read tokens    :          0  ← 90% off
  Regular input tokens :        312
  Output tokens        :      2,104
──────────────────────────────────────────────────────────────
  Total time           :     28.43s
  Actual cost          :   $0.18923
  Without caching      :   $0.20112
  Saved by caching     :   $0.01189  (5.9%)
══════════════════════════════════════════════════════════════
```

---

## Interview probes — answers

### Schema reasoning

`schema_agent.py` crawls using `PRAGMA table_info` — no documentation required. It collects column names, types, sample values, distinct value counts, null percentages, and min/max ranges before sending any of it to Claude. The heuristic FK detection flags the `sku` cross-reference between `po_line_items` and `receipt_lines` as a shared domain key with no constraint enforced — exactly the kind of undocumented join that breaks naive querying.

On ambiguous columns: `is_active` is a flag that could mean anything. The crawl pulls all distinct values and Claude labels it as a vendor deactivation flag — but the evaluation step then scores its own output and flags the gap: *"no deactivation reason or offboarding workflow is documented."* Rather than assuming the annotation is correct, Claude finds its own blind spots and stores the score in the JSON for downstream agents to weight confidence accordingly.

### Retrieval strategy

The unit of knowledge is a joined invoice summary, not a raw row. A raw `invoices` row has seven fields and no vendor name, receipt status, or GL context. The joined summary embeds sentences like `"3-way match complete: NO — missing receipt or GL"` which sits close to match-failure queries in vector space even when the exact words differ.

On precision vs recall: pure semantic search has high recall but low precision for AP queries with explicit structure. The SQL pre-filter pass handles precision by narrowing the candidate set first; semantic ranking handles recall within that set. Cosine distance above `1.2` triggers a low-confidence flag — in financial queries a confident wrong answer is worse than an honest "I don't know."

### Financial domain awareness

AP data is sensitive for two reasons: it controls outbound cash flow, and it's one of the highest-fraud-risk areas in enterprise finance. A **3-way match** is the core AP control — a payment should only be approved when a Purchase Order (authorising the spend), a Goods Receipt (confirming delivery), and an Invoice (requesting payment) all agree on vendor, items, and amounts.

The GL checks (`GL-001`, `AP-001`) catch amounts on the P&L with no backing journal entries — these fail any external audit and break the trial balance at month-end close.

### Anomaly instincts

The spec asked for four rules. Eight were implemented across five categories:

- `AP-002` — the more common AP fraud vector is approving payment before confirming delivery
- `AP-003` — workflow stalls trigger late payment penalties; AP aging reports depend on this
- `VND-001` — open invoices from inactive vendors are either an offboarding failure or a fraud risk
- `GL-001` — imbalanced entries (debits ≠ credits) indicate a corrupted journal that will fail reconciliation

### AI leverage

The LLM is used for reasoning, not boilerplate:

- **`llm_annotate`** — Claude infers business meaning from raw column names and sample values without any human documentation. A regex or lookup table can't do this.
- **`evaluate_schema_context`** — Claude acts as a quality gate, scoring its own output 0–100. The annotation is only trusted downstream if the score is high enough.
- **`query()` synthesis** — schema-aware RAG: the schema summary from `schema_context.json` is included in every synthesis prompt so Claude understands the business meaning of what it's reading.

### Stretch goals — all shipped

| Stretch goal | Status | Where |
|---|---|---|
| Knowledge Graph | Done | `knowledge_graph.py` |
| Synthetic Data Generator | Done | `seed_db.py` (parameterisable) |
| Context Window Packing | Done | `context_packer.py` |

---

## Project structure

```
p2p_context_engine/
    app.py                        Streamlit UI
    rag_pipeline.py               Hybrid RAG pipeline
    schema_agent.py               Schema crawler + LLM annotator
    anomaly_agent.py              8-rule anomaly detector
    context_packer.py             Token-budget context assembler  [stretch]
    knowledge_graph.py            networkx entity dependency graph [stretch]
    cache_logger.py               Token + cost logger
    logger.py                     Shared logging setup
    seed_db.py                    Database seeder + eval data generator [stretch]
    models.py                     Shared dataclasses (Anomaly)
    requirements.txt              anthropic chromadb streamlit python-dotenv networkx
    p2p.db                        SQLite database (generated)
    schema_context.json           Semantic layer (generated)
    anomaly_report.json           Anomaly report (generated)
    crawl_schema_cache.json       Crawl cache (generated)
    llm_annotate_cache.json       LLM annotation cache (generated)
    chroma_db/                    ChromaDB vector index (generated)
    logs/                         Per-script log files (generated)
```

---

## Key design decisions

**Unit of knowledge** — Not a raw SQL row but a joined invoice summary (vendor + PO + receipt status + GL). Dramatically improves retrieval precision for AP questions that span multiple tables.

**Hybrid search** — SQL pre-filtering narrows the candidate set; vector search ranks by relevance within that set. Pure semantic search has high recall but low precision for structured AP queries.

**Prompt caching** — The schema prompt and semantic layer are sent as cached system messages (`cache_control: ephemeral`). The evaluation call reads from cache at 90% cost reduction.

**Low-confidence detection** — ChromaDB cosine distance > 1.2 signals weak retrieval. Claude is explicitly instructed to surface uncertainty rather than hallucinate.

**Self-evaluating schema context** — Claude scores its own annotation 0–100 and flags accuracy risks. The score is stored in `schema_context.json` so downstream agents can weight confidence accordingly.

**Priority-ranked context packing** — The context packer guarantees that the most financially critical information (invoice core, PO, receipts, GL) is always present even if truncated, while lower-signal context (anomaly flags, vendor history) is dropped gracefully when the token budget runs out.
