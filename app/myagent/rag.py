import os
import re
import uuid
import json
import ast
import logging
import importlib
from typing import List, Dict, Optional, Any
from math import isclose

from benchmark.run_benchmark import normalize_text


SentenceTransformer = None
faiss = None
pipeline = None

try:
    from sympy import (
        symbols, Symbol, integrate, diff, simplify, Eq, solve, sympify, S,
        linsolve, Matrix, nsimplify, N
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application
    )
except Exception:
    # SymPy not available -> math features will be disabled; fallbacks used
    parse_expr = None
    sympify = None
    symbols = Symbol = integrate = diff = simplify = Eq = solve = linsolve = nsimplify = N = None

_transformations = (standard_transformations + (implicit_multiplication_application,)) if parse_expr is not None else None

# -------------------- duckduckgo_search lazy import --------------------
_ddg_mod = None
try:
    _ddg_mod = importlib.import_module("duckduckgo_search")
except Exception:
    _ddg_mod = None

# -------------------- transformers pipeline lazy import --------------------
try:
    pipeline = importlib.import_module("transformers").pipeline
except Exception:
    pipeline = None

# -------------------- Config --------------------
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.70))
KB_PATH = os.getenv("KB_PATH", os.path.join(os.path.dirname(__file__), "..", "sample_kb.json"))

logger = logging.getLogger("myagent.rag")
logger.setLevel(logging.INFO)

def _normalize_input(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\u2212", "-")  # unicode minus to hyphen
    s = s.replace("^", "**")
    s = s.replace("²", "**2").replace("³", "**3").replace("⁴", "**4")
    s = re.sub(r'(?<=\d),(?=\d)', '', s)  # remove thousand separators
    s = re.sub(r'\s+', ' ', s)
    return s

def _safe_parse(expr: str):
    if parse_expr is not None:
        try:
            return parse_expr(expr, transformations=_transformations)
        except Exception:
            pass
    # fallback to sympify (may raise)
    return sympify(expr)

def _to_plain(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

# -------------------- Math solver (SymPy-first) --------------------
def math_solver(query: str) -> Optional[Dict[str, Any]]:
    """
    SymPy-first math solver: integrals, derivatives, simplify/evaluate,
    single equation and system solving. Returns structured dict or None.
    """
    if parse_expr is None or sympify is None:
        return None

    raw = (query or "").strip()
    if not raw:
        return None

    q = _normalize_input(raw)
    ql = q.lower()

    contains_eq = "=" in q
    has_var = bool(re.search(r'[A-Za-z_]', q))
    explicit_solve = bool(re.search(r'\bsolve\b|\bfind the roots\b|\bsolve for\b', ql))
    is_integral = bool(re.search(r'\bintegral\b|\bintegrate\b|∫', ql))
    is_derivative = bool(re.search(r'\bderivative\b|\bdifferentiate\b|\bd/d\b', ql))
    is_expr_only = bool(re.match(r'^[\d\s\.\+\-\*\/\^\(\)\[\],]+$', q))
    is_eval = (bool(re.search(r'\bevaluate\b|\bcompute\b|\bwhat is\b', ql) and re.search(r'[0-9\+\-\*\/\^\(\)]', ql))) or is_expr_only

    is_solve = explicit_solve or (contains_eq and has_var)

    try:
        # Solve / Systems
        if is_solve:
            eqs = []
            # split into possible system parts
            parts = re.split(r'\s*;\s*|\n|(?:\s+and\s+)|\s*,\s*', q)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1 and any('=' in p for p in parts):
                for p in parts:
                    if '=' not in p:
                        continue
                    left, right = p.split('=', 1)
                    left_p = _safe_parse(left.strip())
                    right_p = _safe_parse(right.strip())
                    eqs.append(Eq(left_p, right_p))
            else:
                if '=' in q:
                    left, right = q.split('=', 1)
                    left_p = _safe_parse(left.strip())
                    right_p = _safe_parse(right.strip())
                    eqs = [Eq(left_p, right_p)]
                else:
                    m = re.search(r'solve\s+(.+)', ql)
                    if m:
                        expr = m.group(1).strip()
                        if '=' in expr:
                            left, right = expr.split('=', 1)
                            left_p = _safe_parse(left.strip()); right_p = _safe_parse(right.strip())
                            eqs = [Eq(left_p, right_p)]
                        else:
                            left_p = _safe_parse(expr); eqs = [Eq(left_p, 0)]
                    else:
                        eqs = []

            if not eqs:
                return None

            syms = sorted({s for e in eqs for s in e.free_symbols}, key=lambda s: str(s))
            if not syms:
                # numeric equality
                try:
                    diffv = simplify(eqs[0].lhs - eqs[0].rhs)
                    steps = [f"Evaluate equality {str(eqs[0])}.", f"Simplified difference -> {str(diffv)}"]
                    return {"request_id": "sympy-" + str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(diffv), "sources": ["sympy://equality_check"], "confidence": 0.99}
                except Exception:
                    return None

            if len(eqs) == 1 and len(syms) == 1:
                var = syms[0]
                sols = solve(eqs[0], var)
                final = sols[0] if isinstance(sols, (list, tuple)) and len(sols) == 1 else sols
                steps = [f"Equation: {str(eqs[0])}.", f"Solve for {str(var)} -> {str(final)}."]
                return {"request_id": "sympy-" + str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(final), "sources": ["sympy://solve_equation"], "confidence": 0.99}

            # multi-equation: try linsolve then generic solve
            try:
                lsol = linsolve(eqs)
                if lsol and len(lsol) > 0:
                    sols_list = list(lsol)
                    steps = [f"System: {', '.join(str(e) for e in eqs)}.", f"Linear solution -> {str(sols_list)}"]
                    return {"request_id": "sympy-" + str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(sols_list), "sources": ["sympy://linsolve"], "confidence": 0.99}
            except Exception:
                pass
            try:
                soln = solve(eqs, list(syms), dict=True)
                steps = [f"System: {', '.join(str(e) for e in eqs)}.", f"Solve -> {str(soln)}"]
                return {"request_id": "sympy-" + str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(soln), "sources": ["sympy://solve_generic"], "confidence": 0.95}
            except Exception:
                logger.exception("solve fallback failed")
                return None

        # Integrals
        if is_integral:
            m = re.search(r'integrat(?:e|ion)\s+(?:of\s+)?(.+?)\s+from\s+([-\w\.\*\^]+)\s+to\s+([-\w\.\*\^]+)', q, flags=re.I)
            if not m:
                m = re.search(r'(.+?)\s+d([a-zA-Z])(?:\s+from\s+([-\w\.\*\^]+)\s+to\s+([-\w\.\*\^]+))', q, flags=re.I)
            if m:
                expr_str = _normalize_input(m.group(1).strip())
                var = m.group(2) if len(m.groups()) >= 2 else 'x'
                parsed = _safe_parse(expr_str)
                bounds = re.search(r'from\s+([-\w\.\*\^]+)\s+to\s+([-\w\.\*\^]+)', q, flags=re.I)
                x = symbols(var)
                if bounds:
                    a = _safe_parse(_normalize_input(bounds.group(1)))
                    b = _safe_parse(_normalize_input(bounds.group(2)))
                    ant = integrate(parsed, x)
                    val = integrate(parsed, (x, a, b))
                    steps = [f"Integrand: {str(parsed)}.", f"Antiderivative: {str(ant)}.", f"Evaluate from {a} to {b} -> {str(simplify(val))}"]
                    return {"request_id": "sympy-" + str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(simplify(val)), "sources": ["sympy://definite_integral"], "confidence": 0.99}
                else:
                    ant = integrate(parsed, x)
                    steps = [f"Integrand: {str(parsed)}.", f"Antiderivative: {str(ant)} + C"]
                    return {"request_id": "sympy-" + str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(ant) + " + C", "sources": ["sympy://indefinite_integral"], "confidence": 0.99}

        # Derivatives
        if is_derivative:
            m = re.search(r'deriv(?:ative)?(?: of)?\s+(.+?)(?:\s+with respect to\s+([a-zA-Z]))?$', q, flags=re.I)
            if not m:
                m = re.search(r'd/d([a-zA-Z])\s+(.+)', q, flags=re.I)
                if m:
                    var = m.group(1); expr_str = m.group(2); mobj = True
                else:
                    mobj = False
            else:
                expr_str = m.group(1); var = m.group(2) if m.group(2) else 'x'; mobj = True
            if mobj:
                parsed = _safe_parse(_normalize_input(expr_str))
                x = symbols(var)
                deriv = diff(parsed, x)
                steps = [f"Function: {str(parsed)}.", f"Derivative d/d{var} -> {str(deriv)}"]
                return {"request_id": "sympy-" + str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(deriv), "sources": ["sympy://derivative"], "confidence": 0.99}

        # Simplify / Evaluate (arithmetic)
        if is_eval or re.match(r'^[\d\s\.\+\-\*\/\^\(\)\[\],]+$', q):
            expr_str = q
            parsed = None
            try:
                parsed = _safe_parse(expr_str)
            except Exception:
                parsed = None
            if parsed is not None:
                simplified = simplify(parsed)
                try:
                    numeric = simplified.evalf()
                    final = str(nsimplify(numeric)) if numeric.is_Number else str(simplified)
                except Exception:
                    final = str(simplified)
                steps = [f"Parse expression: {str(parsed)}.", f"Simplify -> {str(final)}"]
                return {"request_id": "sympy-" + str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(final), "sources": ["sympy://simplify"], "confidence": 0.99}
    except Exception as e:
        logger.exception("math_solver error: %s", e)
        return None

    return None

# -------------------- Answer canonicalization --------------------
def _canonicalize_scalar_expr(s: str):
    """Return canonical SymPy-like string for scalar expression, fallback normalized text."""
    s0 = (s or "").strip()
    if s0 == "":
        return ""
    if sympify is None:
        # SymPy not available -> normalize text
        return _normalize_input(s0).lower()
    try:
        expr = sympify(s0)
        # if numeric try to return rational or simplified exact form
        try:
            if expr.is_Number:
                n = nsimplify(expr) if hasattr(sympify, "__call__") else expr
                return str(n)
        except Exception:
            pass
        return str(expr)
    except Exception:
        # try mild normalization then sympify
        try:
            s1 = s0.replace("^", "**")
            expr = sympify(s1)
            return str(expr)
        except Exception:
            return _normalize_input(s0).lower()

def canonicalize_answer(ans: Any) -> str:
    """
    Canonicalize predicted answer into a consistent string:
     - JSON-like lists parsed & canonicalized element-wise
     - scalar expressions canonicalized with SymPy when available
    """
    if ans is None:
        return ""
    s = str(ans).strip()
    # Try Python literal parse for lists/tuples
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            out = []
            for el in list(v):
                out.append(_canonicalize_scalar_expr(str(el)))
            return "[" + ", ".join(out) + "]"
    except Exception:
        pass
    # fallback to scalar canonicalization
    return _canonicalize_scalar_expr(s)

# -------------------- KB helpers (lightweight) --------------------
_embed_model = None
_faiss_index = None
_docs: List[Dict] = []
_id_to_payload: Dict[int, Dict] = {}

def _get_embed_model():
    global _embed_model, SentenceTransformer
    if _embed_model is not None:
        return _embed_model
    if SentenceTransformer is None:
        try:
            SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer
        except Exception:
            raise RuntimeError("sentence-transformers not installed")
    _embed_model = SentenceTransformer(EMB_MODEL)
    return _embed_model

def _load_kb():
    global _docs, _id_to_payload
    if _docs:
        return _docs
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = []
    docs = []
    for doc in raw:
        docs.append({"question": doc.get("question",""), "final_answer": doc.get("final_answer",""), "steps": doc.get("steps",[])})
    _docs = docs
    _id_to_payload = {i: d for i, d in enumerate(_docs)}
    return _docs

def build_faiss_index():
    global _faiss_index
    if faiss is None or SentenceTransformer is None:
        # try lazy imports
        try:
            import faiss
        except Exception:
            faiss_local = None
        try:
            import sentence_transformers
        except Exception:
            pass
    _load_kb()
    if not _docs:
        _faiss_index = None
        return
    model = _get_embed_model()
    texts = [d["question"] + "\n" + d["final_answer"] for d in _docs]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = __import__("numpy").linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    vectors = vectors / norms
    dim = vectors.shape[1]
    import faiss as _faiss
    index = _faiss.IndexFlatIP(dim)
    index.add(vectors.astype("float32"))
    _faiss_index = index

def search_kb(query: str, top_k: int = 3):
    global _faiss_index
    if faiss is None or SentenceTransformer is None:
        return []
    if _faiss_index is None:
        build_faiss_index()
    if _faiss_index is None:
        return []
    model = _get_embed_model()
    qv = model.encode([query], convert_to_numpy=True)[0]
    import numpy as _np
    qv = qv / (_np.linalg.norm(qv) + 1e-12)
    D, I = _faiss_index.search(_np.array([qv], dtype="float32"), top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0: continue
        payload = _id_to_payload.get(int(idx), {})
        hits.append({"score": float(score), "payload": payload, "doc_id": int(idx)})
    return hits

# -------------------- Web (duckduckgo) wrapper --------------------
def mcp_search_and_extract(query: str, top_k: int = 5) -> List[Dict]:
    results = []
    if not _ddg_mod:
        return results
    try:
        if hasattr(_ddg_mod, "ddg"):
            raw = _ddg_mod.ddg(query, max_results=top_k) or []
            for r in raw:
                results.append({"title": r.get("title",""), "url": r.get("href") or r.get("url",""), "snippet": r.get("body") or r.get("snippet","")})
            return results
        if hasattr(_ddg_mod, "search"):
            raw = _ddg_mod.search(query, max_results=top_k) or []
            for r in raw:
                results.append({"title": r.get("title",""), "url": r.get("url",""), "snippet": r.get("snippet","")})
            return results
        if hasattr(_ddg_mod, "ddg_answers"):
            try:
                ans = _ddg_mod.ddg_answers(query)
                candidates = []
                for key in ("answers","related","results"):
                    if isinstance(ans, dict) and key in ans and isinstance(ans[key], list):
                        candidates = ans[key][:top_k]
                        break
                for a in candidates:
                    if isinstance(a, dict):
                        results.append({"title": a.get("title",""), "url": a.get("href") or a.get("url",""), "snippet": a.get("body") or a.get("snippet","")})
                return results
            except Exception:
                pass
    except Exception as e:
        logger.exception("mcp_search_and_extract error: %s", e)
    return results

# -------------------- HF fallback synthesize (lazy) --------------------
_hf_gen = None
def _get_hf_generator():
    global _hf_gen, pipeline
    if _hf_gen is not None:
        return _hf_gen
    if pipeline is None:
        try:
            pipeline = importlib.import_module("transformers").pipeline
        except Exception:
            pipeline = None
    if pipeline is None:
        raise RuntimeError("transformers not available")
    _hf_gen = pipeline("text2text-generation", model=HF_MODEL, device_map=None)
    return _hf_gen

def synthesize_with_hf(query: str, sources: List[Dict]) -> Dict:
    try:
        gen = _get_hf_generator()
    except Exception as e:
        logger.exception("HF unavailable: %s", e)
        return {"steps": ["HF_UNAVAILABLE"], "final_answer": "", "sources": [], "confidence": 0.0}
    src_text = ""
    for s in (sources or [])[:5]:
        src_text += f"- {s.get('title','')}\n  {s.get('url','')}\n  {s.get('snippet','')}\n"
    prompt = (
        "You are a math professor. Answer step-by-step and then give a final concise answer.\n\n"
        f"Question: {query}\n\n"
        f"Sources:\n{src_text}\n\n"
        "Return as:\nSTEPS:\n1. ...\nFINAL: one-line final answer\n"
    )
    try:
        out = gen(prompt, max_new_tokens=256, do_sample=False)[0].get("generated_text","")
    except Exception as e:
        logger.exception("HF generation failed: %s", e)
        return {"steps": ["GEN_FAILED"], "final_answer": "", "sources": [], "confidence": 0.0}
    steps = []
    final = "See steps"
    if "STEPS:" in out and "FINAL:" in out:
        try:
            body = out.split("STEPS:",1)[1]
            parts = body.split("FINAL:",1)
            step_text = parts[0].strip()
            final = parts[1].strip()
            for line in step_text.splitlines():
                line = line.strip()
                if not line: continue
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                steps.append(line)
        except Exception:
            steps = [out.strip()]
    else:
        steps = [out.strip()]
    return {"steps": steps, "final_answer": final, "sources": [s.get("url") for s in (sources or [])], "confidence": 0.6}

# -------------------- Top-level handler --------------------
def handle_query(query: str, requester: str = "anon") -> Dict[str, Any]:
    """
    Improved pipeline that:
      1) uses SymPy math_solver if possible (already canonicalized)
      2) KB lookup (faiss) if threshold passed
      3) Web + HF fallback
    Additional: ALWAYS attempt to extract a machine-readable final_answer:
      - prefer existing final_answer
      - else try to parse the last step for an expression/number
      - use sympy to canonicalize
    """
    def _extract_from_steps_or_text(text_or_steps):
        """
        Accepts either steps list or text, try to find a math expression/number to canonicalize.
        Returns canonicalized string or empty.
        """
        cand = ""
        # if steps list, join them; else use string
        if isinstance(text_or_steps, list):
            joined = " ".join([str(s) for s in text_or_steps if s])
        else:
            joined = str(text_or_steps or "")
        joined = joined.strip()
        if not joined:
            return ""
        # heuristics: look for fraction or expression like a/b or numbers or something with x, pi, E, I
        # try to find last line that contains "=" or "->" or looks like "final" or "answer"
        # common patterns produced by HF: "FINAL: <answer>"
        m_final = re.search(r'final[:\-\s]*([\s\S]+)$', joined, flags=re.I)
        if m_final:
            cand = m_final.group(1).strip()
        else:
            # find last line with digits or math operators
            lines = [ln.strip() for ln in joined.splitlines() if ln.strip()]
            # check lines from bottom up
            for ln in reversed(lines):
                if re.search(r'[\d\w\)\]\}\.]+[\s]*[=\+\-\*\/\^\)]?', ln):
                    cand = ln
                    break
            if not cand:
                cand = joined  # last resort: try whole text

        # Remove leading labels like "Answer:", "Final:" etc.
        cand = re.sub(r'^(answer|ans|final|result|=>|->)[:\s\-]*', '', cand, flags=re.I).strip()

        # Attempt to extract a math-like substring (first reasonable token chunk)
        # look for bracketed lists or parentheses or numeric expression
        m_list = re.search(r'(\[[^\]]+\]|\([^\)]+\)|\{[^\}]+\})', cand)
        if m_list:
            cand = m_list.group(1)
        else:
            # look for a reasonably short expression near end (up to 200 chars)
            cand = cand[-200:] if len(cand) > 200 else cand

        # Last attempt: try to sympify
        try:
            c = _canonicalize_scalar_expr(cand)
            if c:
                return c
        except Exception:
            pass

        # fallback normalization
        return normalize_text(cand)

    # 1) SymPy deterministic solver
    try:
        math_res = math_solver(query)
        if math_res is not None:
            if math_res.get("final_answer", None) not in (None, ""):
                try:
                    math_res["final_answer"] = canonicalize_answer(math_res["final_answer"])
                except Exception:
                    pass
            else:
                # attempt to derive final from steps
                steps = math_res.get("steps") or []
                if steps:
                    fin = _extract_from_steps_or_text(steps)
                    if fin:
                        math_res["final_answer"] = fin
            return math_res
    except Exception:
        logger.exception("math_solver crashed; will fallback")

    # 2) KB lookup
    try:
        hits = search_kb(query, top_k=3)
        if hits:
            top = hits[0]
            score = float(top.get("score", 0.0))
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                payload = top.get("payload", {})
                res = {
                    "request_id": f"kb-{top.get('doc_id')}",
                    "steps": payload.get("steps", []),
                    "final_answer": payload.get("final_answer", ""),
                    "sources": [f"kb://{payload.get('original_id') or top.get('doc_id')}"],
                    "confidence": 0.95
                }
                # canonicalize or extract from steps
                if not res.get("final_answer"):
                    res["final_answer"] = _extract_from_steps_or_text(res.get("steps", []))
                else:
                    try:
                        res["final_answer"] = canonicalize_answer(res["final_answer"])
                    except Exception:
                        pass
                return res
    except Exception:
        logger.exception("KB lookup failed; continuing")

    # 3) Web search + HF fallback
    sources = mcp_search_and_extract(query, top_k=5)
    synthesis = synthesize_with_hf(query, sources)
    synthesis["request_id"] = "mcp-" + (requester or "anon")

    # If HF returned steps or final_answer, try to extract/canonicalize a machine final_answer
    fa = synthesis.get("final_answer", None)
    steps = synthesis.get("steps", None)
    final_candidate = ""
    if fa and str(fa).strip() not in ("", "See steps", "See steps."):
        # try canonicalize directly
        try:
            final_candidate = canonicalize_answer(fa)
        except Exception:
            # try extraction heuristics
            final_candidate = _extract_from_steps_or_text(fa)
    else:
        # try from steps
        final_candidate = _extract_from_steps_or_text(steps or [])

    if final_candidate:
        synthesis["final_answer"] = final_candidate
    else:
        # keep what exists (may be prose); append human-review note
        synthesis.setdefault("steps", []).append("NOTE: Unable to extract a concise machine answer; human review recommended.")
        synthesis.setdefault("final_answer", "")

    try:
        if synthesis.get("final_answer", None):
            synthesis["final_answer"] = canonicalize_answer(synthesis["final_answer"])
    except Exception:
        pass

    return synthesis

class RouterAgent:
    def __init__(self, *args, **kwargs):
        pass
    def handle_query(self, query: str, requester: str = None):
        return handle_query(query, requester=requester)
    def ask(self, query: str, requester: str = None):
        return self.handle_query(query, requester=requester)
    def route(self, query: str, requester: str = None):
        return self.handle_query(query, requester=requester)

# -------------------- CLI quick self-check --------------------
if __name__ == "__main__":
    print("RAG module quick self-check")
    print("SymPy available:", parse_expr is not None)
    try:
        print("math:", math_solver("Integrate x^2 from 0 to 1"))
        print("arith:", math_solver("39+71"))
        print("eq:", math_solver("x + 3 = -2"))
        print("system:", math_solver("x+y=3; x-y=1"))
    except Exception as e:
        print("self-test error:", e)
