"""
schema_agent.py — Crawls the P2P SQLite schema, infers business meaning
via Claude, and writes schema_context.json.

Purpose:
    This script performs "schema archaeology" — given only a database connection,
    it reverse-engineers the business meaning of every table and column using an
    LLM, then saves the result as a structured JSON semantic layer.

    The semantic layer is used downstream by:
        - rag_pipeline.py  — to ground LLM answers in schema context
        - context_packer.py — to assemble invoice context for AI agents

    Three stages:
        1. crawl_schema  — extract raw schema from SQLite (no LLM)
        2. llm_annotate  — send schema to Claude for semantic annotation
        3. evaluate      — ask Claude to self-score the annotation quality

    Both stage 1 and stage 2 outputs are cached to disk so they don't need
    to be re-run on every execution. Use --force to bypass the cache.

Usage:
    python schema_agent.py --db p2p.db --out schema_context.json
    python schema_agent.py --force          # re-run LLM annotation
    python schema_agent.py --skip-eval      # skip self-evaluation step
"""

from dotenv import load_dotenv
load_dotenv()
from logger import get_logger
log = get_logger("schema_agent")
import sqlite3, json, argparse, textwrap
import anthropic
import os
#model_name = 


# ── llm_annotate ─────────────────────────────────────────────────────────────

def llm_annotate(raw_schema: dict, client: anthropic.Anthropic) -> dict:
    """
    Send the raw schema (produced by crawl_schema) to Claude and ask it to
    produce a semantic annotation for every table and column.

    This is the core "schema archaeology" step — Claude infers business meaning
    from column names, types, and sample values without any human documentation.

    Parameters
    ----------
    raw_schema : dict
        Output of crawl_schema() — contains tables, columns, row counts,
        sample values, domain values, col_stats, and implicit FKs.
    client : anthropic.Anthropic
        Authenticated Anthropic API client with prompt caching enabled.

    Returns
    -------
    dict
        Semantic layer with tables, relationships, domain glossary, and
        query patterns — ready to be written to schema_context.json.

    Prompt caching:
        The full prompt is sent as a cached system message (cache_control:
        ephemeral). On the first call, Claude writes the cache (higher cost).
        On subsequent calls (e.g. evaluate_schema_context), Claude reads from
        cache at 90% discount. The user message is a single "x" placeholder
        because all meaningful content is in the system prompt.
    """

    # Build a trimmed version of the schema to send to the LLM.
    # We exclude keys that start with "_" (internal metadata like _implicit_fks)
    # because they're not table definitions and would confuse Claude.
    schema_for_llm = {}
    for table, meta in raw_schema.items():
        # Skip internal metadata keys — they're for our use, not Claude's
        if table.startswith("_"):
            continue

        # Build a per-table dict with the fields Claude needs for annotation.
        # We include sample_values as a flat dict (col_name → list of strings)
        # rather than the raw row list, which is easier for Claude to interpret.
        schema_for_llm[table] = {
            "columns":    meta["columns"],     # column names, types, PK flags
            "row_count":  meta["row_count"],   # total rows — gives Claude scale
            "sample_values": {
                # For each column, collect up to 5 sample values as strings.
                # str() handles None, int, float uniformly.
                # r.get(col["name"], "") returns "" if the column is missing from a row.
                col["name"]: [str(r.get(col["name"], "")) for r in meta["samples"]]
                for col in meta["columns"]
            }
        }

    # Import the prompt builder from the helpers subpackage.
    # get_prompt() constructs the full annotation prompt using the schema data
    # and the implicit FK relationships detected by crawl_schema.
    from schema_agent_helpers.prompt import get_prompt
    prompt = get_prompt(schema_for_llm, raw_schema)

    # Wrap the prompt as a cached system message.
    # cache_control: ephemeral tells Anthropic to cache this content for ~5 minutes.
    # This means if evaluate_schema_context() runs shortly after llm_annotate(),
    # it reads the same cached prompt at 90% cost reduction.
    system = [
        {
            "type":          "text",
            "text":          prompt,
            "cache_control": {"type": "ephemeral"},  # enable prompt caching
        }
    ]

    # Call the Claude API.
    # model: claude-opus-4-5 — most capable model for complex schema reasoning
    # max_tokens: 8192 — large output budget; schema annotations can be verbose
    # messages: single user turn with "x" placeholder — all content is in system
    # system: the cached prompt built above
    from schema_agent_helpers.cache_report import extract_usage, save_usage_report
    import time

    # use time to mark start of LLM call
    start = time.perf_counter()
    message = client.messages.create(
        model='claude-opus-4-5',
        max_tokens=8192,
        messages=[{"role": "user", "content": "x"}],
        system=system,
    )
    # save time when LLM call ends
    elapsed = time.perf_counter() - start
    # extract usage to determine amount of cached tokens from prompt
    usage   = extract_usage(message)

    # create cache prompt report
    save_usage_report(
    usage_data          = usage,
    elapsed_seconds     = elapsed,
    documents_processed = 3,   # however many you processed this run
    skipped             = 0,
    failed              = 0,
    model               = 'claude-opus-4-5',
    json_path           = "schema_agent_helpers/token_usage_log.json",
    txt_path            = "schema_agent_helpers/token_usage_report.txt",
    )
   
    # Extract the text content from the first (and only) content block.
    # .strip() removes leading/trailing whitespace and newlines.
    raw_text = message.content[0].text.strip()

    # Claude sometimes wraps JSON in markdown code fences (```json ... ```).
    # If the response starts with ```, strip the opening fence line and the
    # closing ``` to get clean JSON for json.loads().
    if raw_text.startswith("```"):
        # Split on the first newline to remove the opening fence (e.g. "```json")
        # then rsplit on the last ``` to remove the closing fence
        raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0]

    # Parse the cleaned JSON string into a Python dict.
    # This will raise json.JSONDecodeError if the response was truncated
    # (e.g. if max_tokens was too low) — the caller handles that exception.
    semantic = json.loads(raw_text)

    # Attach row counts from the original crawl to each table in the semantic layer.
    # Claude doesn't always preserve these in its output, so we inject them
    # from the raw schema to ensure downstream scripts have accurate counts.
    for table, meta in semantic["tables"].items():
        if table in raw_schema:
            meta["row_count"] = raw_schema[table]["row_count"]

    # Attach the raw implicit FK list from the crawler so downstream tools
    # can see both the LLM-inferred relationships and the heuristic ones.
    semantic["_implicit_fks_raw"] = raw_schema.get("_implicit_fks", [])

    return semantic


# ── evaluate_schema_context ───────────────────────────────────────────────────

def evaluate_schema_context(semantic: dict, client: anthropic.Anthropic) -> dict:
    """
    Ask Claude to self-evaluate the quality of the schema context it produced.

    This is a "self-critique" step — Claude reviews its own annotation and
    scores it for coverage and accuracy, then flags any missing relationships,
    accuracy risks, and recommended improvements.

    Parameters
    ----------
    semantic : dict
        The output of llm_annotate() — the full semantic layer JSON.
    client : anthropic.Anthropic
        Authenticated Anthropic API client with prompt caching enabled.

    Returns
    -------
    dict
        Evaluation result with keys:
            overall_score          : int (0-100)
            coverage_score         : int (0-100)
            accuracy_risks         : list[str]
            missing_relationships  : list[str]
            recommended_improvements: list[str]

    Why self-evaluation matters:
        Without this step, we have no signal on whether the semantic layer is
        accurate enough to trust for financial queries. A score below ~80 means
        the annotation has gaps that could cause the RAG pipeline to hallucinate
        or miss important relationships.

    Prompt caching:
        The full semantic JSON is large. Sending it as a cached system message
        means if this function is called multiple times in the same session,
        subsequent calls read from cache at 90% cost reduction.
    """

    # Build the evaluation prompt using textwrap.dedent() to strip the
    # leading whitespace caused by the indentation of the multiline string.
    # The full semantic dict is embedded as pretty-printed JSON so Claude
    # can read every table, column, and relationship it produced.
    prompt = textwrap.dedent(f"""
    You produced this schema_context.json for a P2P system.
    Rate its quality for use in financial AI queries.

    Schema context:
    {json.dumps(semantic, indent=2)}

    Return JSON only:
    {{
      "overall_score": 0-100,
      "coverage_score": 0-100,
      "accuracy_risks": ["<risk 1>", "<risk 2>"],
      "missing_relationships": ["<description>"],
      "recommended_improvements": ["<improvement>"]
    }}
    """)

    # Wrap the evaluation prompt as a cached system message — same pattern
    # as llm_annotate(). If the semantic content is the same as the previous
    # call, Anthropic may serve this from the prompt cache at reduced cost.
    system = [
        {
            "type":          "text",
            "text":          prompt,
            "cache_control": {"type": "ephemeral"},  # enable prompt caching
        }
    ]

    msg = client.messages.create(
        model='claude-opus-4-5',
        max_tokens=8192,
        messages=[{"role": "user", "content": "x"}],
        system=system
    )

    # Extract the raw text response and strip whitespace
    raw = msg.content[0].text.strip()

    # Strip markdown code fences if Claude wrapped the JSON in them
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    # Parse and return the evaluation dict.
    # Caller is responsible for handling json.JSONDecodeError if truncated.
    return json.loads(raw)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    """
    CLI entry point for schema_agent.py.

    Orchestrates the three-stage pipeline:
        1. crawl_schema  — load from cache or re-crawl the database
        2. llm_annotate  — load from cache or re-run Claude annotation
        3. evaluate      — score the annotation (skippable with --skip-eval)

    Both stage 1 and stage 2 are cached to disk as JSON files. This means
    on a normal run (no --force flags), only the evaluation step calls the API.
    The cache files are:
        crawl_schema_cache.json  — raw schema from SQLite
        llm_annotate_cache.json  — Claude's semantic annotation

    CLI arguments:
        --db         Path to the SQLite database (default: p2p.db)
        --out        Path for the final schema_context.json (default: schema_context.json)
        --force      Re-run both crawl and LLM annotation, ignoring caches
        --skip-eval  Skip the self-evaluation API call (faster, cheaper)
    """

    # Set up the argument parser with all supported CLI flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",        default="p2p.db",              help="SQLite DB path")
    parser.add_argument("--out",       default="schema_context.json", help="Output JSON path")
    parser.add_argument("--force",     action="store_true",           help="Re-run LLM even if file exists")
    parser.add_argument("--skip-eval", action="store_true",           help="Skip self-evaluation")

    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Create the Anthropic client with the prompt caching beta header.
    # The beta header is required to use cache_control in system messages.
    # Without it, the cache_control field is silently ignored and you pay
    # full price for every API call.
    client = anthropic.Anthropic(
        api_key=api_key,
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )

    CRAWL_CACHE = "crawl_schema_cache.json"   # output of crawl_schema()
    LLM_CACHE   = "llm_annotate_cache.json"   # output of llm_annotate()

    # ── Step 1: crawl_schema ──────────────────────────────────────────────────

    # Try to load the crawl cache from disk first.
    # If the file doesn't exist or is malformed, fall through to the except block
    # and re-crawl the database.
    # Using a bare except: here catches all exceptions (file not found, JSON error, etc.)
    # — acceptable for a caching pattern where any failure means "re-run".
    try:
        log.info(f"📂 Loading cached crawl_schema from {CRAWL_CACHE}")

        # Open and parse the cache file
        with open(CRAWL_CACHE) as f:
            raw = json.load(f)

        # DEBUG-level log so this detail only appears in the log file, not the console
        log.debug(f"Loaded {len(raw)} keys from crawl cache")

    except:
        # Cache miss or error — import and run crawl_schema from the helpers subpackage.
        # The import is inside the except block so it's only loaded when needed.
        from schema_agent_helpers.crawl_schema import crawl_schema

        log.info("🔍 Crawling schema…")

        # Run the schema crawler against the database
        raw = crawl_schema(args.db)

        # Count tables (exclude keys starting with "_" which are metadata)
        table_count = len([t for t in raw if not t.startswith("_")])

        # Count implicit FK relationships detected by the heuristic
        fk_count = len(raw.get("_implicit_fks", []))

        log.info(f"   Found {table_count} tables, {fk_count} implicit FK relationships")

        # Log the full table name list at DEBUG level (too verbose for console)
        log.debug(f"Tables found: {[t for t in raw if not t.startswith('_')]}")

        # Save the crawl result to disk so next run can skip this step
        with open(CRAWL_CACHE, "w") as f:
            json.dump(raw, f, indent=2)

        log.info(f"💾 Saved crawl_schema output to {CRAWL_CACHE}")

    # ── Step 2: llm_annotate ──────────────────────────────────────────────────

    # Same try/except caching pattern as Step 1.
    # Try to load the LLM annotation from disk; fall through to re-run if missing.
    try:
        log.info(f"📂 Loading cached llm_annotate from {LLM_CACHE}")

        # Open and parse the LLM annotation cache
        with open(LLM_CACHE) as f:
            semantic = json.load(f)

        # Log table count at DEBUG level
        log.debug(f"Loaded semantic layer with {len(semantic.get('tables', {}))} tables")

    except:
        # Cache miss — run llm_annotate() which calls the Claude API
        log.info("🧠 Running llm_annotate via Claude…")

        try:
            # Call Claude to annotate the raw schema.
            # This is the expensive step — it sends the full schema to the API.
            semantic = llm_annotate(raw, client)

            log.info(f"   Annotated {len(semantic.get('tables', {}))} tables")
            log.info(f"   Found {len(semantic.get('relationships', []))} relationships")

            # Save the annotation to disk for future runs
            with open(LLM_CACHE, "w") as f:
                json.dump(semantic, f, indent=2)

            log.info(f"💾 Saved llm_annotate output to {LLM_CACHE}")

        except json.JSONDecodeError as e:
            # JSON parse failure usually means the response was truncated
            # because max_tokens was too low. Log clearly and re-raise.
            log.error(f"❌ JSON parse failed — response likely truncated: {e}")
            log.debug("Tip: increase max_tokens in llm_annotate()")
            raise

        except Exception as e:
            # Any other failure (network error, auth error, etc.)
            log.error(f"❌ llm_annotate failed: {e}")
            raise

    # ── Step 3: evaluate ──────────────────────────────────────────────────────

    # Run self-evaluation unless --skip-eval was passed.
    # Skipping is useful when iterating quickly and the API call cost matters.
    if not args.skip_eval:
        log.info("✅ Running self-evaluation…")

        try:
            # Ask Claude to score the semantic layer it produced.
            # Returns a dict with overall_score, coverage_score, accuracy_risks, etc.
            evaluation = evaluate_schema_context(semantic, client)

            # Attach the evaluation results to the semantic dict so they're
            # saved alongside the annotation in schema_context.json
            semantic["_evaluation"] = evaluation

            # Extract individual scores for logging
            o_score = evaluation.get("overall_score", "?")
            c_score = evaluation.get("coverage_score", "?")

            # Log scores at INFO level so they appear on the console
            log.info(f"   Overall score: {o_score}/100")
            log.info(f"   Coverage score {c_score}/100")

            # Log each accuracy risk as a WARNING — visible on console and in file
            for risk in evaluation.get("accuracy_risks", []):
                log.warning(f"   ⚠ {risk}")

            # Log each missing relationship as a WARNING with a flag emoji
            for rel in evaluation.get("missing_relationships", []):
                log.warning(f"   🚩 {rel}")

            # Log each recommended improvement as a WARNING with a wrench emoji
            for impr in evaluation.get("recommended_improvements", []):
                log.warning(f"   🔧 {impr}")

            # Log the full evaluation JSON at DEBUG level (file only, not console)
            log.debug(f"Full evaluation: {json.dumps(evaluation, indent=2)}")

        except Exception as e:
            # Evaluation failure is non-fatal but we re-raise so the caller
            # knows something went wrong — don't silently swallow errors.
            log.error(f"❌ Evaluation failed: {e}")
            raise

    else:
        # --skip-eval was passed — note it in the log and move on
        log.info("⏭ Skipping evaluation (--skip-eval)")

    # ── Write final output ────────────────────────────────────────────────────

    # Write the final schema_context.json — the semantic layer with evaluation
    # scores attached. This is the file consumed by rag_pipeline.py and
    # context_packer.py.
    with open(args.out, "w") as f:
        json.dump(semantic, f, indent=2)

    log.info(f"✅ schema_context.json written → {args.out}")

    # Visual separator in the log file to make it easy to spot session boundaries
    log.info("=" * 62)


# ── Entry point ───────────────────────────────────────────────────────────────

# Only call main() when this script is run directly (python schema_agent.py).
# When schema_agent is imported by another module (e.g. in tests), main()
# is not called automatically.
if __name__ == "__main__":
    main()
