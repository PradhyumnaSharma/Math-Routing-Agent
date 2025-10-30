import os, json, re
from dotenv import load_dotenv
load_dotenv()

GROQ_ENABLED = os.getenv("GROQ_ENABLED", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.0"))

def _extract_json_from_text(text):
    m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        txt = m.group(1).replace("'", '"')
        txt = re.sub(r",\s*}", "}", txt)
        txt = re.sub(r",\s*]", "]", txt)
        try:
            return json.loads(txt)
        except Exception:
            return None

def synthesize_with_groq(question: str, sources: list):
    if not GROQ_ENABLED:
        return {"steps": ["Groq adapter not enabled (GROQ_ENABLED=false)."], "final_answer": "", "sources": [], "confidence": 0.0}
    if not GROQ_API_KEY:
        return {"steps": ["Groq API key not set (GROQ_API_KEY)."], "final_answer": "", "sources": [], "confidence": 0.0}

    src_text = ""
    for s in (sources or []):
        src_text += f"- {s.get('title','')}\n  {s.get('url','')}\n  snippet: {s.get('snippet','')}\n"
    prompt = (
        "You are a precise mathematics professor. Given the question and optional sources, "
        "produce a step-by-step solution suitable for a student. If sources are present, cite them inline.\n\n"
        f"Question: {question}\n\nSources:\n{src_text}\n\n"
        "Output: Return a JSON object only, with keys:\n"
        " - steps: array of short step strings (ordered)\n"
        " - final_answer: concise final answer (string)\n"
        " - brief_sources: list of source urls used\n"
        " - confidence: number between 0.0 and 1.0\n\n"
        "Return only the JSON object in your response (no additional prose)."
    )

    try:
        from groq import Groq
        client = Groq(temperature=GROQ_TEMPERATURE, groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_NAME)
    except Exception:
        return {
            "steps": [
                "Groq SDK not installed or not importable. Please install the vendor ChatGroq SDK or enable HF fallback.",
                "Adapter attempted to import 'chatgroq' but failed."
            ],
            "final_answer": "",
            "sources": [s.get("url") for s in (sources or [])],
            "confidence": 0.0
        }

    possible_methods = [
        ("generate", {"prompt": prompt, "max_tokens": 1024}),
        ("chat", {"prompt": prompt, "max_tokens": 1024}),
        ("create_completion", {"prompt": prompt, "max_tokens": 1024}),
        ("complete", {"prompt": prompt, "max_tokens": 1024}),
    ]

    raw_text = None
    for method_name, kwargs in possible_methods:
        fn = getattr(client, method_name, None)
        if not callable(fn):
            continue
        try:
            resp = fn(**kwargs)
            if isinstance(resp, str):
                raw_text = resp
            elif isinstance(resp, dict):
                raw_text = resp.get("text") or resp.get("output") or resp.get("content") or json.dumps(resp)
            else:
                raw_text = str(resp)
            break
        except TypeError:
            try:
                resp = fn(prompt)
                raw_text = resp if isinstance(resp, str) else (json.dumps(resp) if isinstance(resp, dict) else str(resp))
                break
            except Exception:
                continue
        except Exception:
            continue

    if raw_text is None:
        return {
            "steps": [
                "Groq SDK found but could not call any known method. Check SDK docs or implement correct call in adapter."
            ],
            "final_answer": "",
            "sources": [s.get("url") for s in (sources or [])],
            "confidence": 0.0
        }

    parsed = _extract_json_from_text(raw_text)
    if parsed:
        steps = parsed.get("steps", []) or ([parsed.get("explanation", "")] if parsed.get("explanation") else [])
        final_ans = parsed.get("final_answer", parsed.get("answer", ""))
        brief_sources = parsed.get("brief_sources", [s.get("url") for s in (sources or [])])
        confidence = float(parsed.get("confidence", 0.0))
        return {"steps": steps, "final_answer": final_ans, "sources": brief_sources, "confidence": confidence}

    return {
        "steps": [raw_text.strip()[:4000]],
        "final_answer": "See steps",
        "sources": [s.get("url") for s in (sources or [])],
        "confidence": 0.6
    }
