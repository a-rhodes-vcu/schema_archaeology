"""
rag_pipeline.py — Hybrid RAG over P2P transactional data.

Purpose:
    Enables natural language querying over 5000+ invoices by combining two
    complementary retrieval strategies:

    1. SQL pre-filtering  — narrows the candidate set using structured filters
                            (invoice number, payment terms, status) extracted
                            from the question. Fast, precise, zero false positives.

    2. Semantic search    — ranks the filtered candidates by meaning using
                            ChromaDB vector embeddings. Catches questions that
                            SQL alone can't answer (e.g. "which invoices look risky?")

    Together these form "hybrid search" — the SQL pass boosts precision,
    the semantic pass provides recall for unstructured questions.

    The retrieved chunks are then synthesised by Claude into a grounded,
    financial-domain answer. Claude is explicitly instructed not to hallucinate —
    if the context doesn't support the answer, it must say so.

Key design choices:
    Unit of knowledge  = a joined invoice summary (vendor + PO + receipt + GL),
                         not a raw SQL row. Raw rows lose all relationships.
    Vector store       = ChromaDB PersistentClient (survives restarts).
                         Swap for Qdrant/pgvector by changing the client only.
    Hybrid search      = SQL pre-pass + ChromaDB semantic similarity ranking.
    LLM synthesis      = Claude with explicit grounding instructions.

Usage:
    python rag_pipeline.py --db p2p.db --context schema_context.json
    # Then type natural language questions at the interactive prompt.
"""


import sqlite3, json, argparse, textwrap, sys

from typing import Optional

# chromadb — vector database for embedding and retrieving invoice chunks
# PersistentClient stores the index to disk (./chroma_db/) so it survives restarts
import chromadb

# embedding_functions — provides the DefaultEmbeddingFunction which uses
# the sentence-transformers/all-MiniLM-L6-v2 model to embed text into vectors
from chromadb.utils import embedding_functions

import anthropic

from dotenv import load_dotenv
load_dotenv()

# os — used to read ANTHROPIC_API_KEY from the environment after load_dotenv()
import os

# The name of the ChromaDB collection where invoice chunks are stored.
# Using a constant avoids typos when the name is referenced in multiple places.
COLLECTION_NAME = "p2p_invoice_chunks"


# ── build_invoice_chunks ──────────────────────────────────────────────────────

def build_invoice_chunks(db_path: str) -> list[dict]:
    """
    Query the P2P database and build one "chunk" per invoice, where each chunk
    is a natural language summary of that invoice joined with its vendor, PO,
    receipt status, and GL summary.

    This is the core "unit of knowledge" decision — the most important design
    choice in the entire RAG pipeline.

    Why not embed raw SQL rows?
        A raw invoices row only has: id, vendor_id, po_id, invoice_number,
        amount, status, due_date. That's 7 fields with no context about the
        vendor name, whether goods were received, or whether GL was posted.
        A semantic search over raw rows would have very poor precision for
        AP questions like "which invoices failed 3-way match?"

    Why a joined natural language summary?
        The embedding model encodes meaning, not just keywords. A sentence like
        "Goods receipt on file: no. 3-way match complete: NO — missing receipt"
        embeds close to questions about 3-way match failures, even if the exact
        words differ. Joined summaries give the embedding model rich context.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (e.g. "p2p.db").

    Returns
    -------
    list[dict]
        One dict per invoice, each with keys:
            id   : str  — invoice ID as string (ChromaDB requires string IDs)
            text : str  — natural language summary for embedding
            meta : dict — structured fields for ChromaDB metadata filtering
    """

    # Open the database connection
    con = sqlite3.connect(db_path)

    # sqlite3.Row enables column-name access on result rows (r["amount"] vs r[0])
    con.row_factory = sqlite3.Row

    # Create a cursor for executing the main chunk-building query
    cur = con.cursor()

    # The core chunk query — joins 5 tables to build a complete AP view per invoice.
    # Each JOIN is explained below:
    #
    # JOIN vendors v         — get vendor name, payment terms, credit limit, category
    # JOIN purchase_orders   — get PO status and creation date
    # LEFT JOIN goods_receipts — LEFT because not all POs have receipts (that's an anomaly)
    # LEFT JOIN (subquery)   — aggregate GL entries per invoice; LEFT because some invoices
    #                          have no GL entries (AP-001 anomaly)
    #
    # CASE WHEN gr.id IS NOT NULL THEN 'yes' ELSE 'no' END
    #   → 'yes' if at least one goods receipt exists for this PO, 'no' otherwise
    #   → IS NOT NULL works because LEFT JOIN sets gr.id = NULL when no match exists
    #
    # COALESCE(gl.total_debit, 0) — returns 0 if gl is NULL (no GL entries for this invoice)
    #   → prevents NULL values in the text representation
    #
    # LIMIT 5000 — caps at 5000 invoices to match our seed data size
    rows = cur.execute("""
        SELECT
            i.id              AS invoice_id,
            i.invoice_number,
            i.amount,
            i.status          AS invoice_status,
            i.due_date,
            v.name            AS vendor_name,
            v.payment_terms,
            v.credit_limit,
            v.category        AS vendor_category,
            po.status         AS po_status,
            po.created_at     AS po_date,
            -- 3-way match: receipt exists?
            CASE WHEN gr.id IS NOT NULL THEN 'yes' ELSE 'no' END AS receipt_exists,
            -- GL posted?
            CASE WHEN gl.invoice_id IS NOT NULL THEN 'yes' ELSE 'no' END AS gl_posted,
            COALESCE(gl.total_debit, 0)  AS gl_debit,
            COALESCE(gl.total_credit, 0) AS gl_credit
        FROM invoices i
        JOIN vendors v  ON v.id  = i.vendor_id
        JOIN purchase_orders po ON po.id = i.po_id
        LEFT JOIN goods_receipts gr ON gr.po_id = i.po_id
        LEFT JOIN (
            SELECT invoice_id,
                   SUM(debit)  AS total_debit,
                   SUM(credit) AS total_credit
            FROM gl_entries GROUP BY invoice_id
        ) gl ON gl.invoice_id = i.id
        LIMIT 5000
    """).fetchall()
    # fetchall() returns all 5000 rows as a list — acceptable here because we
    # need all chunks to build the vector index. In production with millions of
    # invoices, you'd batch this or use a streaming cursor.

    # Accumulate the chunk dicts in a list
    chunks = []

    for r in rows:
        # Convert sqlite3.Row to plain dict for standard dict access
        r = dict(r)

        # Build the natural language summary that will be embedded.
        # textwrap.dedent() removes the leading spaces from the indented f-string.
        # .strip() removes the leading/trailing newlines.
        #
        # Every field is expressed as a complete sentence rather than key: value,
        # because sentence-transformers embed sentences better than structured data.
        # "Goods receipt on file: yes" embeds more meaningfully than "receipt_exists: yes".
        text = textwrap.dedent(f"""
            Invoice {r['invoice_number']} (ID {r['invoice_id']}) from {r['vendor_name']}.
            Amount: ${r['amount']:,.2f}. Status: {r['invoice_status']}.
            Due: {r['due_date']}. Payment terms: {r['payment_terms']}.
            Vendor category: {r['vendor_category']}.
            Vendor credit limit: ${r['credit_limit']:,.2f}.
            Purchase order status: {r['po_status']} (created {r['po_date']}).
            Goods receipt on file: {r['receipt_exists']}.
            GL entry posted: {r['gl_posted']}.
            GL debit total: ${r['gl_debit']:,.2f}. GL credit total: ${r['gl_credit']:,.2f}.
            3-way match complete: {'yes' if r['receipt_exists'] == 'yes' and r['gl_posted'] == 'yes' else 'NO — missing receipt or GL'}.
        """).strip()
        # The 3-way match line explicitly states the conclusion rather than leaving
        # it for the LLM to infer — this improves retrieval precision for 3-way match questions.

        # Build the chunk dict with three keys:
        # id   — ChromaDB requires string IDs; invoice IDs are ints so we convert
        # text — the natural language summary that gets embedded into a vector
        # meta — structured metadata stored alongside the vector for WHERE filtering
        chunks.append({
            "id":   str(r["invoice_id"]),
            "text": text,
            "meta": {
                # Metadata fields are used for ChromaDB WHERE clause filtering.
                # Only include fields that callers will actually filter on —
                # ChromaDB metadata has a size limit per document.
                "invoice_id":     r["invoice_id"],      # int — for $in filter
                "invoice_number": r["invoice_number"],  # str — for display
                "invoice_status": r["invoice_status"],  # str — for status filter
                "vendor_name":    r["vendor_name"],     # str — for display
                "payment_terms":  r["payment_terms"],   # str — for terms filter
                "amount":         r["amount"],           # float — for range filter
                "receipt_exists": r["receipt_exists"],  # "yes"/"no" — for 3-way match filter
                "gl_posted":      r["gl_posted"],        # "yes"/"no" — for GL filter
                "due_date":       r["due_date"],         # str — for date filter
            }
        })

    # Close the database connection — releases the file lock
    con.close()

    # Return the full list of chunk dicts ready for ChromaDB indexing
    return chunks


# ── sql_filter_invoice_ids ────────────────────────────────────────────────────

def sql_filter_invoice_ids(db_path: str, filter_sql: str) -> list[str]:
    """
    Execute a narrow SQL query to get candidate invoice IDs for the hybrid
    search pre-filter pass.

    This is the "SQL half" of hybrid search. By narrowing the candidate set
    before semantic search, we improve precision — semantic search only ranks
    invoices that are structurally relevant (right vendor, right status, etc.)
    rather than all 5000 invoices.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    filter_sql : str
        A SQL query that returns a single column named 'invoice_id'.
        Built by _detect_filter() based on the user's question.

    Returns
    -------
    list[str]
        Invoice IDs as strings (ChromaDB uses string IDs).
        Empty list if the query returns no results or fails.
    """

    try:
        # Open a fresh connection for this filter query
        con = sqlite3.connect(db_path)

        # Execute the filter SQL and extract the first column (invoice_id) from each row.
        # str(r[0]) converts int IDs to strings to match ChromaDB's string ID format.
        ids = [str(r[0]) for r in con.execute(filter_sql).fetchall()]

        # Close the connection immediately after fetching
        con.close()

        return ids

    except sqlite3.OperationalError as e:
        # SQL syntax error or missing table — log and return empty list so
        # the caller falls back to full semantic search rather than crashing
        print(f"   ⚠ SQL filter failed ({e}) — falling back to full search")
        return []


# ── get_or_build_collection ───────────────────────────────────────────────────

def get_or_build_collection(chunks: list[dict]) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection from disk, or build and index a new one
    from the provided chunks if it doesn't exist yet.

    This is the persistence layer — using PersistentClient means the vector
    index survives process restarts. The first run (building) takes ~1 minute
    for 5000 chunks; every subsequent run (loading) takes ~1 second.

    Parameters
    ----------
    chunks : list[dict]
        Output of build_invoice_chunks() — used only when building a new index.

    Returns
    -------
    chromadb.Collection
        The loaded or newly created ChromaDB collection, ready for querying.
    """

    # PersistentClient stores the index in the ./chroma_db/ directory.
    # On first run: creates the directory and initialises the SQLite+vector store.
    # On subsequent runs: loads the existing index from disk without re-embedding.
    client = chromadb.PersistentClient(path="./chroma_db")

    # DefaultEmbeddingFunction uses sentence-transformers/all-MiniLM-L6-v2
    # — a lightweight but effective model for semantic similarity.
    # It runs locally (no API call) and produces 384-dimensional embeddings.
    ef = embedding_functions.DefaultEmbeddingFunction()

    try:
        # Try to load the existing collection by name.
        # Raises an exception if the collection doesn't exist yet.
        col = client.get_collection(COLLECTION_NAME, embedding_function=ef)

        # Collection found — report how many documents are already indexed
        print(f"  📦 Loaded existing collection ({col.count()} docs)")

    except Exception:
        # Collection doesn't exist — build and index it from scratch

        # Create a new empty collection with the given name and embedding function
        col = client.create_collection(COLLECTION_NAME, embedding_function=ef)
        print(f"  🔨 Building index over {len(chunks)} invoice chunks…")

        # Insert chunks in batches of 500 to avoid ChromaDB's per-call size limit.
        # ChromaDB has an internal limit of ~5461 documents per add() call,
        # but 500 is a safer and more memory-efficient batch size.
        batch = 500
        for i in range(0, len(chunks), batch):
            # Slice the chunks list to get the current batch
            b = chunks[i:i+batch]

            # Add the batch to ChromaDB.
            # ChromaDB embeds the documents automatically using the ef function.
            # ids       — unique string identifiers (invoice IDs as strings)
            # documents — the natural language text that gets embedded into vectors
            # metadatas — structured fields stored alongside the vector for filtering
            col.add(
                ids=[c["id"] for c in b],
                documents=[c["text"] for c in b],
                metadatas=[c["meta"] for c in b],
            )

        # Report the total number of indexed documents
        print(f"  ✅ Indexed {col.count()} chunks")

    # Return the collection object (either loaded or newly built)
    return col


# ── P2PRAG class ──────────────────────────────────────────────────────────────

class P2PRAG:
    """
    The main RAG (Retrieval-Augmented Generation) pipeline for P2P data.

    Combines hybrid search (SQL pre-filter + ChromaDB semantic search) with
    Claude synthesis to answer natural language questions about invoices,
    vendors, and AP data.

    Attributes
    ----------
    db_path : str
        Path to the SQLite database — used for pre-filter queries.
    schema_ctx : dict
        Loaded schema_context.json — provides business context for the synthesis prompt.
    client : anthropic.Anthropic
        Authenticated Anthropic API client for Claude synthesis calls.
    collection : chromadb.Collection
        The ChromaDB collection of embedded invoice chunks — used for semantic search.
    """

    def __init__(self, db_path: str, schema_context_path: str):
        """
        Initialise the RAG pipeline by loading the schema context, setting up
        the Anthropic client, building invoice chunks, and loading/building
        the ChromaDB collection.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database (e.g. "p2p.db").
        schema_context_path : str
            Path to the schema_context.json file produced by schema_agent.py.
        """

        # Store the database path as an instance variable — used by _detect_filter()
        # and query() to run SQL pre-filter queries
        self.db_path = db_path

        # Load the schema context JSON from disk.
        # This provides Claude with the business meaning of each table and column
        # so the synthesis prompt is grounded in domain knowledge.
        with open(schema_context_path) as f:
            self.schema_ctx = json.load(f)

        # Read the API key from the environment (injected by load_dotenv() above)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        # Create the Anthropic client with the prompt caching beta header.
        # The beta header enables cache_control in messages — without it,
        # the caching is silently ignored and you pay full price per call.
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

        # Build invoice chunks from the database — one chunk per invoice with
        # joined vendor, PO, receipt, and GL context (the "unit of knowledge")
        chunks = build_invoice_chunks(db_path)

        # Load the existing ChromaDB index or build a new one if it doesn't exist.
        # This is the most time-consuming step on the first run (~1 minute).
        self.collection = get_or_build_collection(chunks)

    def _detect_filter(self, question: str) -> Optional[str]:
        """
        Analyse the user's question and return a SQL filter query if structured
        filters can be extracted, or None if no filter is applicable.

        This is the "SQL half" of hybrid search. It uses simple pattern matching
        rather than an LLM to keep latency low and avoid an extra API call.

        The filter narrows the candidate set to structurally relevant invoices
        before semantic search ranks them by meaning. This dramatically improves
        precision for questions with explicit filters (payment terms, status, etc.)

        Parameters
        ----------
        question : str
            The user's natural language question.

        Returns
        -------
        Optional[str]
            A SQL query returning a single 'invoice_id' column, or None if no
            structured filter could be extracted from the question.

        Examples
        --------
        "What is INV-2291?" → "SELECT id AS invoice_id FROM invoices WHERE invoice_number = 'INV-2291'"
        "NET60 exposure?"   → "SELECT i.id AS invoice_id FROM invoices i JOIN vendors v ... WHERE v.payment_terms = 'NET60'"
        "approved invoices" → "SELECT id AS invoice_id FROM invoices WHERE status = 'approved' LIMIT 200"
        "risky vendors?"    → None (no structured filter — full semantic search)
        """

        # Lowercase the question for case-insensitive matching
        q_lower = question.lower()

        # ── Pattern 1: Specific invoice number reference ──────────────────────
        # Matches "INV-2291", "inv 2291", "INV2291" etc. using a regex.
        # re is imported inside the method to avoid a module-level import
        # that would be used only in this one place.
        import re

        # re.search scans the entire string for the pattern (vs re.match which
        # only checks the start). re.IGNORECASE makes it case-insensitive.
        # Pattern: "inv" followed by optional "-" or space, then one or more digits.
        m = re.search(r"inv[-\s](\d+)", question, re.IGNORECASE)
        if m:
            # m.group(1) extracts the captured digit group (e.g. "2291")
            return f"SELECT id AS invoice_id FROM invoices WHERE invoice_number = 'INV-{m.group(1)}'"

        # ── Pattern 2: Payment terms filter ───────────────────────────────────
        # Checks if any payment term string appears in the lowercased question.
        # Returns invoices for vendors with those payment terms via a JOIN.
        for term in ["NET60", "NET30", "NET90", "NET15", "COD"]:
            if term.lower() in q_lower:
                # Self-contained query with explicit aliases to avoid SQLite
                # ambiguity errors that caused the original pipeline bug.
                return (
                    f"SELECT i.id AS invoice_id "
                    f"FROM invoices i "
                    f"JOIN vendors v ON v.id = i.vendor_id "
                    f"WHERE v.payment_terms = '{term}'"
                )

        # ── Pattern 3: Invoice status filter ──────────────────────────────────
        # Checks if any status keyword appears in the question.
        # LIMIT 200 prevents returning all invoices for common statuses like
        # "approved" (which could be thousands of rows).
        for status in ["approved", "pending", "disputed", "overdue", "paid"]:
            if status in q_lower:
                return f"SELECT id AS invoice_id FROM invoices WHERE status = '{status}' LIMIT 200"

        # No structured filter found — return None to signal full semantic search
        return None

    def query(self, question: str, n_results: int = 8) -> str:
        """
        Answer a natural language question about the P2P data using hybrid
        retrieval + Claude synthesis.

        Four-step process:
            1. SQL pre-filter  — narrow candidates using structured filter (if any)
            2. Semantic search — rank candidates by meaning using ChromaDB
            3. Confidence check — detect low-confidence retrieval by cosine distance
            4. LLM synthesis   — Claude answers using only the retrieved context

        Parameters
        ----------
        question : str
            The user's natural language question (e.g. "Which vendors are over their credit limit?")
        n_results : int
            Number of chunks to retrieve from ChromaDB (default 8).
            More chunks = more context but longer prompts and slower responses.

        Returns
        -------
        str
            Claude's grounded answer, citing specific invoice numbers and amounts.
        """

        # Print the question for user feedback during interactive mode
        print(f"\n🔍 Query: {question}")

        # ── Step 1: SQL pre-filter (hybrid search) ────────────────────────────

        # Attempt to extract a structured filter from the question
        filter_sql = self._detect_filter(question)

        # Initialise the ChromaDB WHERE clause filter to None (no filter = search all)
        where_ids = None

        if filter_sql:
            # Run the SQL filter to get candidate invoice IDs
            ids = sql_filter_invoice_ids(self.db_path, filter_sql)

            if ids:
                # Build the ChromaDB WHERE clause using the $in operator.
                # $in matches documents whose invoice_id metadata field is in the list.
                # Capped at 500 IDs to avoid ChromaDB performance issues with large lists.
                where_ids = {"invoice_id": {"$in": [int(i) for i in ids[:500]]}}
                # int(i) converts string IDs back to int to match the metadata type
                print(f"   SQL pre-filter: {len(ids)} candidate invoices")
            else:
                # SQL filter returned no results — fall back to full semantic search
                print("   SQL pre-filter returned no results — using full search")

        # ── Step 2: Semantic search ───────────────────────────────────────────

        # Query ChromaDB for the most semantically similar chunks.
        # query_texts is wrapped in a list because ChromaDB supports batch queries
        # (multiple questions at once) — we only need one here.
        results = self.collection.query(
            query_texts=[question],
            # min() prevents requesting more results than exist in the collection
            n_results=min(n_results, self.collection.count()),
            # where=None means search all chunks; where=dict narrows to filtered IDs
            where=where_ids,
        )

        # results["documents"] is a list of lists (one per query).
        # [0] extracts the results for our single query.
        docs = results["documents"][0]      # list of chunk text strings
        metas = results["metadatas"][0]     # list of metadata dicts
        distances = results["distances"][0] # list of cosine distances (lower = more similar)

        # ── Step 3: Confidence check ──────────────────────────────────────────

        # Cosine distance > 1.2 indicates weak semantic similarity — the retrieved
        # chunks may not be relevant to the question. If ALL results have high
        # distance, retrieval has failed and Claude should say so rather than guess.
        # all() returns True only if every distance exceeds the threshold.
        low_confidence = all(d > 1.2 for d in distances)

        if low_confidence:
            # Log the distances so the user can diagnose retrieval quality
            print("   ⚠ Low-confidence retrieval — distances:", [round(d,3) for d in distances])

        # Join all retrieved chunk texts with a separator for the synthesis prompt.
        # "---" separates chunks visually so Claude can distinguish between invoices.
        context_block = "\n\n---\n\n".join(docs)

        # ── Step 4: LLM synthesis (grounded answer) ───────────────────────────

        # Build a concise schema summary from the loaded schema_context.json.
        # We include only the business_purpose of each table (not full column details)
        # to keep the prompt focused and avoid token waste.
        # Dict comprehension: {table_name: business_purpose} for each table in schema
        schema_summary = json.dumps({
            k: v.get("business_purpose","")
            for k, v in self.schema_ctx.get("tables", {}).items()
        }, indent=2)

        # Build the synthesis prompt.
        # Key instructions:
        #   "Answer ONLY using the retrieved context" — prevents hallucination
        #   "If context is insufficient, say so" — explicit uncertainty instruction
        #   "Flag any data quality concerns" — encourages anomaly surfacing
        # The low_confidence warning is conditionally included if retrieval was poor.
        prompt = textwrap.dedent(f"""
        You are an AP (Accounts Payable) analyst assistant. Answer ONLY using the
        retrieved context below. If the context is insufficient, say so explicitly —
        do NOT hallucinate data.

        Schema reference:
        {schema_summary}

        Retrieved invoice context:
        {context_block}

        {"⚠ NOTE: Retrieval confidence is low — explicitly state uncertainty in your answer." if low_confidence else ""}

        Question: {question}

        Answer concisely with specific numbers and invoice/vendor names from the context.
        Flag any data quality concerns you notice.
        """)

        # Call Claude to synthesise an answer from the retrieved context.
        # model: claude-opus-4-5 — best reasoning for financial domain questions
        # max_tokens: 1024 — enough for a detailed answer without excessive cost
        # messages: single user turn with the full retrieval + synthesis prompt
        msg = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the text from the first content block of Claude's response
        answer = msg.content[0].text

        # Print the answer to the console for interactive mode feedback
        print(f"\n💬 Answer:\n{answer}\n")

        # Return the answer string (used by app.py Streamlit UI and test harness)
        return answer


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    """
    CLI entry point for rag_pipeline.py.

    Initialises the P2PRAG pipeline, runs three preset questions
    automatically, then enters an interactive loop where the user can type
    natural language questions until they press Ctrl+C.

    CLI arguments:
        --db       Path to the SQLite database (default: p2p.db)
        --context  Path to schema_context.json (default: schema_context.json)
    """

    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",      default="p2p.db")
    parser.add_argument("--context", default="schema_context.json")

    # Parse the CLI arguments from sys.argv
    args = parser.parse_args()

    # Initialise the full RAG pipeline — loads schema, sets up Anthropic client,
    # builds invoice chunks, and loads/builds the ChromaDB collection.
    rag = P2PRAG(args.db, args.context)

    # The three questions specified in the brief.
    # These are run automatically on startup to demonstrate the pipeline works.
    test_questions = [
        "Which vendors have invoices that were approved without a complete goods receipt?",
        "What's the total AP exposure for vendors on NET60 terms?",
        "Are there any invoices from the same vendor with duplicate invoice numbers?",
    ]

    # Visual separator for the preset question section
    print("\n" + "="*60)
    print("Running preset questions")
    print("="*60)

    # Run each test question through the full RAG pipeline
    for q in test_questions:
        rag.query(q)

    # ── Interactive mode ──────────────────────────────────────────────────────

    # After the preset questions, drop into an interactive prompt where the user
    # can type any question and get an immediate answer.
    print("\nInteractive mode — type your questions (Ctrl+C to exit):")

    while True:
        try:
            # input() blocks until the user presses Enter.
            # .strip() removes leading/trailing whitespace from the input.
            q = input("\n> ").strip()

            # Only call query() if the user typed something (not just Enter)
            if q:
                rag.query(q)

        except KeyboardInterrupt:
            # Ctrl+C raises KeyboardInterrupt — catch it for a clean exit
            # rather than showing a Python traceback to the user.
            print("\nDone.")

            # sys.exit(0) exits with status code 0 (success).
            # Without this, the while loop would continue after the except block.
            sys.exit(0)


# ── Entry point ───────────────────────────────────────────────────────────────

# Only call main() when this script is run directly (python rag_pipeline.py).
# When P2PRAG is imported by other scripts (e.g. app.py), main() is not
# called — only the class and functions are available for import.
if __name__ == "__main__":
    main()
