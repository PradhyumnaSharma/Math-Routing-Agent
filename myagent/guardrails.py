import re
from sympy import sympify, N

PII_PATTERNS = [
    r"\b\d{10}\b",
    r"[\w\.-]+@[\w\.-]+\.\w+"
]

NON_MATH_BLACKLIST = [
    "dating", "sex", "hack", "give me account", "buy", "subscribe"
]

def run_input_guardrails(query: str):
    q = query.lower()
    for p in PII_PATTERNS:
        if re.search(p, query):
            raise ValueError("Query contains PII; blocked by input guardrails.")
    for bad in NON_MATH_BLACKLIST:
        if bad in q:
            raise ValueError("Query contains non-educational content; only math allowed.")
    return True

def run_output_guardrails(result: dict):
    if not isinstance(result, dict):
        raise ValueError("Result format invalid")
    if "final_answer" not in result or "steps" not in result:
        raise ValueError("Result missing required fields")
    fa = str(result.get("final_answer",""))
    try:
        expr = sympify(fa)
        val = N(expr)
        result["confidence"] = float(result.get("confidence", 0.5))
    except Exception:
        result["confidence"] = float(result.get("confidence", 0.5))
    return True
