from dotenv import load_dotenv
load_dotenv()
from logger import get_logger

import anthropic
import os


api_key = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(
        api_key=api_key,
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )


def find_customer_anomaly():
    prompt = textwrap.dedent(f"""


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

