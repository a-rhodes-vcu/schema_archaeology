"""
app.py — Streamlit UI for the P2P RAG pipeline.

Purpose:
    Provides a browser-based chat interface for querying the P2P database
    using natural language. Built on top of rag_pipeline.py — the Streamlit
    layer handles UI state, user input, and result display while P2PRAG
    handles retrieval and synthesis.

    Key Streamlit concepts used:
        st.session_state    — persists data across reruns (chat history, pre-filled questions)
        @st.cache_resource  — caches the P2PRAG object so the vector index loads once
        st.chat_message     — renders chat bubbles for user and assistant turns
        st.chat_input       — the text input bar at the bottom of the page
        st.spinner          — shows a loading indicator while the RAG pipeline runs

    Streamlit reruns the entire script top-to-bottom on every user interaction.
    This means every widget, loop, and conditional is re-evaluated on each rerun.
    st.session_state and @st.cache_resource are the mechanisms for preserving
    state and expensive objects across those reruns.

Run with:
    streamlit run app.py
    uv run streamlit run app.py  (if using uv)
"""

# json — used to serialise the schema context dict into a JSON string for the prompt
import json

# time — time.perf_counter() measures elapsed wall-clock time for the response metric
import time

# streamlit — the web UI framework. st.* functions render UI components.
# Every st.* call adds an element to the page in the order it's called.
import streamlit as st

# P2PRAG — the main RAG pipeline class from rag_pipeline.py.
# Provides hybrid search (SQL pre-filter + ChromaDB semantic) and Claude synthesis.
from rag_pipeline import P2PRAG


# ── Page config ───────────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script — sets browser tab title,
# favicon, and layout. "wide" uses the full browser width instead of a
# narrow centered column.

st.set_page_config(
    page_title="P2P Context Engine",  # browser tab title
    page_icon="🧾",                   # browser tab favicon
    layout="wide",                    # use full browser width
)

# Render the main page title (large H1 heading)
st.title("🧾 P2P Context Engine")

# Render a small muted caption below the title
st.caption("Ask natural language questions about your Purchase-to-Pay data.")


# ── Sidebar config ────────────────────────────────────────────────────────────
# The sidebar is a collapsible panel on the left side of the page.
# All st.* calls inside this context manager render into the sidebar.

with st.sidebar:

    # Section heading for the configuration controls
    st.header("Configuration")

    # Text input for the database path.
    # value= sets the default — user can override it in the UI.
    # The return value is whatever string the user has typed (or the default).
    db_path = st.text_input("Database path", value="p2p.db")

    # Text input for the schema context path produced by schema_agent.py
    context_path = st.text_input("Schema context path", value="schema_context.json")

    # Slider for how many ChromaDB chunks to retrieve per query.
    # More chunks = more context for Claude but longer prompts and slower responses.
    # min_value / max_value define the slider range; value= sets the default.
    n_results = st.slider("Results to retrieve", min_value=3, max_value=20, value=8)

    # Horizontal divider line between sections
    st.divider()

    # Section heading for the example question buttons
    st.subheader("Example questions")

    # List of pre-written AP questions for one-click querying
    examples = [
        "Which vendors have invoices approved without a goods receipt?",
        "What is the total AP exposure for vendors on NET60 terms?",
        "Are there duplicate invoice numbers from the same vendor?",
        "Which invoices are overdue and still pending?",
        "Which vendors are over their credit limit?",
    ]

    for example in examples:
        # Render one button per example question.
        # use_container_width=True stretches the button to fill the sidebar width.
        # st.button() returns True on the rerun triggered by clicking it.
        if st.button(example, use_container_width=True):
            # Store the clicked question in session_state so the chat input
            # below can pre-fill it. session_state persists across reruns.
            st.session_state.question = example

    # Another horizontal divider before the clear button
    st.divider()

    # Button to clear the entire chat history.
    # Resets st.session_state.messages to an empty list, which causes the
    # chat history rendering loop below to render nothing on the next rerun.
    if st.button("🗑 Clear chat history", use_container_width=True):
        st.session_state.messages = []


# ── Load RAG pipeline (cached so it only builds once) ────────────────────────

# @st.cache_resource caches the return value of load_rag() in memory.
# The cache persists across reruns as long as the arguments (db_path, context_path)
# don't change. This is critical because P2PRAG.__init__() builds or loads the
# ChromaDB index — an expensive operation that would take ~1 minute on every rerun
# without caching.
#
# show_spinner= displays a loading message in the UI while the function runs
# on the first call (subsequent calls return instantly from cache).
@st.cache_resource(show_spinner="Loading RAG pipeline and vector index…")
def load_rag(db_path: str, context_path: str) -> P2PRAG:
    """
    Initialise and return a P2PRAG instance.

    Decorated with @st.cache_resource so this only runs once per session,
    not on every Streamlit rerun. The cached P2PRAG object (including its
    ChromaDB collection) is reused on all subsequent reruns.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database — passed through to P2PRAG.__init__().
    context_path : str
        Path to schema_context.json — passed through to P2PRAG.__init__().

    Returns
    -------
    P2PRAG
        A fully initialised RAG pipeline with the ChromaDB index loaded.
    """
    # Initialise the P2PRAG pipeline — this is the expensive step.
    # On first call: builds invoice chunks, loads/creates ChromaDB index.
    # On subsequent calls: returned from cache instantly.
    return P2PRAG(db_path, context_path)


# Attempt to load the RAG pipeline using the paths from the sidebar inputs.
# If either file is missing, show a user-friendly error and stop execution
# rather than crashing with a Python traceback.
try:
    # load_rag() returns the cached P2PRAG on reruns after the first call
    rag = load_rag(db_path, context_path)

except FileNotFoundError as e:
    # Show a red error banner with the specific missing file message
    st.error(f"Could not load pipeline: {e}")

    # Show a blue info box explaining how to fix the issue
    st.info("Make sure you have run `python schema_agent.py` first to generate schema_context.json")

    # st.stop() halts script execution immediately — nothing below this line
    # renders in the UI when the pipeline fails to load.
    st.stop()


# ── Chat history ──────────────────────────────────────────────────────────────

# Initialise the messages list in session_state if it doesn't exist yet.
# This runs on the very first page load. On subsequent reruns, the existing
# list is preserved and this block is skipped.
if "messages" not in st.session_state:
    # Empty list — no messages yet on first load
    st.session_state.messages = []

# Render all previous messages from the chat history.
# This loop re-renders every message on every rerun, which is how Streamlit
# maintains the appearance of a persistent chat — it redraws everything each time.
for msg in st.session_state.messages:

    # st.chat_message() renders a chat bubble with the appropriate avatar.
    # role="user" shows a person icon; role="assistant" shows a bot icon.
    with st.chat_message(msg["role"]):

        # Render the message content as markdown (supports bold, code blocks, etc.)
        st.markdown(msg["content"])

        # If this message has metadata attached (assistant messages do),
        # render it in a collapsible expander below the message.
        if msg.get("meta"):
            with st.expander("Retrieved context"):
                # st.json() renders the dict as formatted, collapsible JSON
                st.json(msg["meta"])


# ── Chat input ────────────────────────────────────────────────────────────────

# Check if a sidebar example button was clicked on this rerun.
# st.session_state.pop("question", "") removes the key and returns its value,
# or returns "" if the key doesn't exist. This ensures the pre-fill only
# applies once — on the rerun triggered by the button click.
default_question = st.session_state.pop("question", "")

# Render the chat input bar at the bottom of the page.
# st.chat_input() returns the submitted text when the user presses Enter,
# or None if no submission occurred on this rerun.
# The `or default_question` means: if chat_input returned None (no submission),
# use the sidebar-button pre-fill instead (if any).
question = st.chat_input(
    "Ask a question about your P2P data…",  # placeholder text in the input bar
) or default_question

# Only run the RAG pipeline if we have a question (either from chat input
# or from a sidebar button click).
if question:

    # Add the user's message to the history list so it persists across reruns
    st.session_state.messages.append({"role": "user", "content": question})

    # Render the user's message as a chat bubble immediately (before the answer)
    with st.chat_message("user"):
        st.markdown(question)

    # Render the assistant's response inside an assistant chat bubble
    with st.chat_message("assistant"):

        # st.spinner() shows an animated loading indicator while the code inside
        # the context manager is running. Hides automatically when done.
        with st.spinner("Searching invoices…"):

            # Record the start time for the response time metric
            # time.perf_counter() is more precise than time.time() for short intervals
            start = time.perf_counter()

            # ── Hybrid search: Step 1 — SQL pre-filter ────────────────────────

            # Attempt to extract a structured SQL filter from the question.
            # Returns a SQL string or None if no filter could be detected.
            filter_sql = rag._detect_filter(question)

            # Initialise the ChromaDB WHERE clause filter to None (search all chunks)
            where_ids = None

            # filter_info will hold a human-readable description of the filter applied,
            # shown below the answer as a caption. None means no filter was applied.
            filter_info = None

            if filter_sql:
                # Import the helper function from rag_pipeline to execute the SQL filter
                from rag_pipeline import sql_filter_invoice_ids

                # Run the SQL filter query to get candidate invoice IDs
                ids = sql_filter_invoice_ids(rag.db_path, filter_sql)

                if ids:
                    # Build the ChromaDB WHERE clause — $in matches documents
                    # whose invoice_id metadata field is in the provided list.
                    # Capped at 500 IDs to avoid ChromaDB performance issues.
                    # int(i) converts string IDs back to int to match metadata type.
                    where_ids = {"invoice_id": {"$in": [int(i) for i in ids[:500]]}}

                    # Build the filter description for display below the answer.
                    # :, formats the number with commas (e.g. 1,234)
                    filter_info = f"SQL pre-filter applied: {len(ids):,} candidate invoices"

            # ── Hybrid search: Step 2 — Semantic search ───────────────────────

            # Query ChromaDB for the most semantically similar invoice chunks.
            # query_texts is a list because ChromaDB supports batch queries —
            # we only need one question at a time here.
            results = rag.collection.query(
                query_texts=[question],
                # min() prevents requesting more results than exist in the collection
                n_results=min(n_results, rag.collection.count()),
                # where=None = search all chunks; where=dict = filter by invoice IDs
                where=where_ids,
            )

            # Extract results for our single query (index [0] of the batch)
            docs      = results["documents"][0]   # list of chunk text strings
            metas     = results["metadatas"][0]   # list of metadata dicts
            distances = results["distances"][0]   # list of cosine distances

            # Confidence check: if ALL distances exceed 1.2, retrieval is weak.
            # Cosine distance > 1.2 means the chunks are not semantically close
            # to the question — Claude will be warned to express uncertainty.
            low_conf = all(d > 1.2 for d in distances)

            # ── Build synthesis prompt ─────────────────────────────────────────

            # Join all retrieved chunk texts with a separator so Claude can
            # distinguish between different invoices in the context block
            context_block = "\n\n---\n\n".join(docs)

            # Build a concise schema summary from schema_context.json.
            # Only include the business_purpose of each table (not full column
            # details) to keep the prompt focused and avoid token waste.
            schema_summary = json.dumps({
                k: v.get("business_purpose", "")   # get purpose or empty string if missing
                for k, v in rag.schema_ctx.get("tables", {}).items()
            }, indent=2)

            # textwrap must be imported here if not already imported at the top.
            # dedent() removes the leading indentation from the multiline f-string.
            import textwrap

            # Build the synthesis prompt with explicit grounding instructions.
            # "Answer ONLY using the retrieved context" prevents hallucination.
            # The low_conf warning is conditionally included using an inline ternary.
            prompt = textwrap.dedent(f"""
            You are an AP (Accounts Payable) analyst assistant. Answer ONLY using the
            retrieved context below. If the context is insufficient, say so explicitly —
            do NOT hallucinate data.

            Schema reference:
            {schema_summary}

            Retrieved invoice context:
            {context_block}

            {"⚠ NOTE: Retrieval confidence is low — explicitly state uncertainty." if low_conf else ""}

            Question: {question}

            Answer concisely with specific numbers and invoice/vendor names from the context.
            Flag any data quality concerns you notice.
            """)

            # ── Call Claude for synthesis ──────────────────────────────────────

            # Call the Anthropic API using the rag object's authenticated client.
            # model: claude-opus-4-5 — best reasoning for financial domain questions
            # max_tokens: 1024 — enough for a detailed answer without excessive cost
            message = rag.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            # Record the end time and calculate elapsed seconds
            elapsed = time.perf_counter() - start

            # Extract the text from the first content block of Claude's response
            answer = message.content[0].text

        # ── Display results (outside the spinner, inside the assistant bubble) ─

        # Render Claude's answer as markdown (supports bold, lists, code, etc.)
        st.markdown(answer)

        # ── Metrics row ───────────────────────────────────────────────────────

        # Create three equal-width columns side by side for the metric cards
        col1, col2, col3 = st.columns(3)

        # Number of chunks retrieved from ChromaDB
        col1.metric("Chunks retrieved", len(docs))

        # Total elapsed time from question to answer, formatted to 1 decimal place
        col2.metric("Time", f"{elapsed:.1f}s")

        # Confidence indicator — "Low ⚠" if all distances > 1.2, "OK ✓" otherwise
        col3.metric("Confidence", "Low ⚠" if low_conf else "OK ✓")

        # Show the SQL filter info as a small caption if a filter was applied
        if filter_info:
            # st.caption() renders small muted text below the metrics
            st.caption(f"🔍 {filter_info}")

        # Show a yellow warning banner if retrieval confidence was low
        if low_conf:
            st.warning("Retrieval confidence is low — answer may be incomplete.")

        # ── Retrieved chunks expander ─────────────────────────────────────────

        # Collapsible section showing the raw retrieved chunks for transparency.
        # Lets users see exactly what context Claude used to form its answer.
        with st.expander("Retrieved invoice chunks"):

            # zip() pairs up docs, metas, and distances by index so we can
            # display them together. enumerate() adds a 1-based counter (i+1).
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):

                # Chunk header showing its rank and cosine distance
                # dist:.3f formats to 3 decimal places (e.g. 0.847)
                st.markdown(f"**Chunk {i+1}** — distance: `{dist:.3f}`")

                # st.text() renders the chunk text as preformatted monospace text
                # (no markdown processing — shows the raw chunk content)
                st.text(doc)

                # st.json() renders the metadata dict as formatted, collapsible JSON
                st.json(meta)

                # Horizontal divider between chunks for visual separation
                st.divider()

    # ── Save assistant response to chat history ───────────────────────────────

    # Append the assistant's response to the messages list so it appears
    # in the chat history on the next rerun.
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,           # the text shown in the chat bubble
        "meta":    {                 # metadata shown in the "Retrieved context" expander
            "distances":      distances,   # list of cosine distances for each chunk
            "low_confidence": low_conf     # bool — whether retrieval was flagged as weak
        }
    })
