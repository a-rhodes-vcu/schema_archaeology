import json
import time

# ── token tracking ─────────────────────────────────────────────────────────────

# Pricing per million tokens (claude-opus-4-5)
PRICING = {
    "input":       15.00,
    "cache_write":  3.75,
    "cache_read":   1.50,
    "output":      75.00,
}

def extract_usage(message) -> dict:
    """Extract all token counts from an Anthropic response."""
    usage = {
        "input_tokens":  message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens":  message.usage.input_tokens + message.usage.output_tokens,
        "cache_read_input_tokens":  0,
        "cache_creation_input_tokens": 0,
    }
    try:
        usage["cache_read_input_tokens"]     = message.usage.cache_read_input_tokens or 0
        usage["cache_creation_input_tokens"] = message.usage.cache_creation_input_tokens or 0
    except AttributeError:
        pass
    return usage


def save_usage_report(
    usage_data:          dict,
    elapsed_seconds:     float,
    documents_processed: int = 0,
    skipped:             int = 0,
    failed:              int = 0,
    model:               str = "claude-opus-4-5",
    json_path:           str = "schema_agent_helpers/token_usage_log.json",
    txt_path:            str = "schema_agent_helpers/token_usage_report.txt",
) -> None:
    """
    Save token usage to JSON and write the formatted session summary to a txt file.
    """

    # ── compute costs ──────────────────────────────────────────────────────────
    p = PRICING

    cache_write = usage_data.get("cache_creation_input_tokens", 0)
    cache_read  = usage_data.get("cache_read_input_tokens", 0)
    regular_in  = usage_data.get("input_tokens", 0)
    output      = usage_data.get("output_tokens", 0)

    actual = (
        cache_write * p["cache_write"] / 1_000_000
        + cache_read  * p["cache_read"]  / 1_000_000
        + regular_in  * p["input"]       / 1_000_000
        + output      * p["output"]      / 1_000_000
    )

    total_input_if_no_cache = cache_write + cache_read + regular_in
    without = (
        total_input_if_no_cache * p["input"]  / 1_000_000
        + output                * p["output"] / 1_000_000
    )

    saved = without - actual
    pct   = (saved / without * 100) if without > 0 else 0.0

    # ── save raw JSON ──────────────────────────────────────────────────────────
    usage_data["actual_cost"]       = round(actual,  5)
    usage_data["cost_without_cache"] = round(without, 5)
    usage_data["cost_saved"]         = round(saved,   5)
    usage_data["elapsed_seconds"]    = round(elapsed_seconds, 2)
    usage_data["model"]              = model

    with open(json_path, "w") as f:
        json.dump(usage_data, f, indent=4)

    # ── build formatted report ────────────────────────────────────────────────
    W = 62  # line width
    border = "═" * W
    divider = "─" * W

    lines = [
        border,
        "  SESSION SUMMARY",
        border,
        f"  Documents processed  : {documents_processed:>10,}",
        f"  Skipped (this run)   : {skipped:>10,}",
        f"  Failed               : {failed:>10,}",
        divider,
        f"  Cache write tokens   : {cache_write:>10,}",
        f"  Cache read tokens    : {cache_read:>10,}",
        f"  Regular input tokens : {regular_in:>10,}",
        f"  Output tokens        : {output:>10,}",
        divider,
        f"  Total time           : {elapsed_seconds:>9.2f}s",
        f"  Actual cost          : ${actual:>10.5f}",
        f"  Without caching      : ${without:>10.5f}",
        f"  Saved by caching     : ${saved:>10.5f}  ({pct:.1f}%)",
        border,
    ]

    report = "\n".join(lines)

    # print to console
    #print(report)

    # write to file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")