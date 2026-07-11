
import  json, textwrap

def get_prompt(schema_for_llm,raw_schema ):
    prompt = textwrap.dedent(f"""
        You are a senior ERP consultant analyzing an undocumented Purchase-to-Pay (P2P) 
        database schema. Your job is to produce a precise semantic layer that an AI agent 
        can use to answer financial questions with zero hallucinations.
        Take your time to understand the schema and the data. It's better to be precise than to be fast.

        Raw schema (tables, columns, sample values):
        {json.dumps(schema_for_llm, indent=2)}

        Implicit relationships detected heuristically:
        {json.dumps(raw_schema.get("_implicit_fks", []), indent=2)}

        Return ONLY a JSON object with this exact structure — no markdown fences, no prose:
        The returned JSON object must be valid JSON and must be parsable with json.loads()
        If it is not possible to create a valid JSON object, return None.
        {{
        "tables": {{
            "<table_name>": {{
            "business_name": "<plain-english name>",
            "business_purpose": "<one sentence: what this table represents>",
            "key_business_entity": "<Vendor | PurchaseOrder | Invoice | GLEntry | etc.>",
            "columns": {{
                "<col_name>": {{
                "business_label": "<plain-english label>",
                "description": "<what this value means in AP/finance context>",
                "sensitive": true | false,
                "domain_values": "<if enum: list known values; else null>"
                }}
            }}
            }}
        }},
        "relationships": [
            {{
            "description": "<one sentence describing the business relationship>",
            "from_table": "...",
            "from_col": "...",
            "to_table": "...",
            "to_col": "...",
            "type": "many-to-one | one-to-many | many-to-many",
            "enforced": true | false,
            "financial_significance": "<why this join matters for AP questions>"
            }}
        ],
        "domain_glossary": {{
            "3-way match": "<definition in context of this schema>",
            "AP exposure": "<definition>",
            "credit limit breach": "<definition>",
            "GL balance": "<definition>"
        }},
        "query_patterns": [
            {{
            "intent": "<business question>",
            "sql_hint": "<skeleton SQL — table names and key JOINs only>"
            }}
        ]
        }}
        """)

    return prompt