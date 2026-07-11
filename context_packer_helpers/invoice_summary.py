from dotenv import load_dotenv
load_dotenv()
import anthropic
import os
import textwrap

api_key = os.environ.get("ANTHROPIC_API_KEY") # reads ANTHROPIC_API_KEY from env
client = anthropic.Anthropic(api_key=api_key, default_headers={"anthropic-beta": "prompt-caching-2024-07-31"})


def claue_invoice_summary(ctx):

    prompt = textwrap.dedent(f"""
    What does this invoice mean?:
    {ctx}
    """)

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}]        
    )
    raw = msg.content[0].text.strip()
    
    return raw