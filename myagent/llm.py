import os, json, re
from dotenv import load_dotenv
load_dotenv()

GROQ_ENABLED = os.getenv("GROQ_ENABLED", "false").lower() == "true"
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")

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

if GROQ_ENABLED:
    from myagent.llm_groq_adapter import synthesize_with_groq as synthesize_solution
else:
    def synthesize_solution(question: str, sources: list):
        try:
            from transformers import pipeline
        except Exception as e:
            return {
                "steps": ["Transformers not installed or failed to import. Install 'transformers' and 'torch'."],
                "final_answer": "",
                "sources": [s.get("url") for s in (sources or [])],
                "confidence": 0.0
            }

        src_text = ""
        for s in sources or []:
            src_text += f"- {s.get('title','')}\n  {s.get('url','')}\n  snippet: {s.get('snippet','')}\n"

        prompt = (
            "You are a careful math professor. Given the question and optional sources, produce a step-by-step solution "
            "suitable for a student. If sources are provided, prefer them and cite them. Return a JSON object with keys: "
            "steps (list), final_answer (string), brief_sources (list), confidence (0-1).\n\n"
            f"Question: {question}\n\nSources:\n{src_text}\n\nJSON output:\n"
        )

        try:
            generator = pipeline("text2text-generation", model=HF_MODEL, device_map="auto")
            out = generator(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        except Exception:
            try:
                generator = pipeline("text-generation", model=HF_MODEL, device_map="auto")
                out = generator(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
            except Exception as e2:
                return {"steps":[f"Model generation failed: {e2}"], "final_answer":"", "sources":[s.get("url") for s in (sources or [])], "confidence":0.0}

        parsed = _extract_json_from_text(out)
        if parsed:
            return {
                "steps": parsed.get("steps", []) or [out],
                "final_answer": parsed.get("final_answer", "") or out,
                "sources": parsed.get("brief_sources", [s.get("url") for s in (sources or [])]),
                "confidence": float(parsed.get("confidence", 0.5))
            }
        return {"steps":[out], "final_answer":"See steps", "sources":[s.get("url") for s in (sources or [])], "confidence":0.6}
