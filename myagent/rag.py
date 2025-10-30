

import os
import json
import uuid
import re
import importlib
import logging
from typing import List, Dict, Optional

import numpy as np

# Embeddings & FAISS (optional; lazy)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

# Transformers (HF) for fallback generation (lazy)
try:
    from transformers import pipeline
except Exception:
    pipeline = None

# SymPy for deterministic math solving
try:
    from sympy import symbols, integrate, diff, simplify, Eq, solve, sympify
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
except Exception:
    symbols = integrate = diff = simplify = Eq = solve = sympify = None
    parse_expr = None

# duckduckgo_search robust import (may expose ddg, search, ddg_answers)
_ddg_mod = None
try:
    _ddg_mod = importlib.import_module("duckduckgo_search")
except Exception:
    _ddg_mod = None

# Configuration (environment-overridable)
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.70))
KB_PATH = os.getenv("KB_PATH", os.path.join(os.path.dirname(__file__), "..", "sample_kb.json"))

# Internal singletons
_embed_model = None
_faiss_index = None
_docs: List[Dict] = []
_id_to_payload: Dict[int, Dict] = {}
_hf_gen = None

# SymPy parsing transforms
_transformations = (standard_transformations + (implicit_multiplication_application,)) if parse_expr is not None else None

logger = logging.getLogger("myagent.rag")
logger.setLevel(logging.INFO)


# --------------------- DuckDuckGo robust wrapper ---------------------
def _normalize_entry(r: dict) -> dict:
    return {
        "title": r.get("title") or r.get("source") or "",
        "url": r.get("href") or r.get("url") or r.get("link") or "",
        "snippet": r.get("body") or r.get("snippet") or r.get("text") or "",
        "trust_score": 0.6
    }


def mcp_search_and_extract(query: str, top_k: int = 5) -> List[Dict]:
    """
    Robust wrapper for duckduckgo_search supporting multiple versions.
    Returns list of {title, url, snippet, trust_score}.
    """
    results = []
    if not _ddg_mod:
        return results
    try:
        if hasattr(_ddg_mod, "ddg"):
            raw = _ddg_mod.ddg(query, max_results=top_k) or []
            for r in raw:
                results.append(_normalize_entry(r))
            return results
        if hasattr(_ddg_mod, "search"):
            raw = _ddg_mod.search(query, max_results=top_k) or []
            for r in raw:
                results.append(_normalize_entry(r))
            return results
        if hasattr(_ddg_mod, "ddg_answers"):
            try:
                ans = _ddg_mod.ddg_answers(query)
                candidates = []
                for key in ("answers", "related", "results"):
                    if isinstance(ans, dict) and key in ans and isinstance(ans[key], list):
                        candidates = ans[key][:top_k]
                        break
                for a in candidates:
                    if isinstance(a, dict):
                        results.append(_normalize_entry(a))
                return results
            except Exception:
                pass
    except Exception as e:
        logger.exception("mcp_search_and_extract failed: %s", e)
        return []
    return results


# --------------------- Embeddings + FAISS (lazy) ---------------------
def _get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    _embed_model = SentenceTransformer(EMB_MODEL)
    return _embed_model


def _load_kb() -> List[Dict]:
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
        payload = {
            "original_id": doc.get("id"),
            "question": doc.get("question"),
            "steps": doc.get("steps", []),
            "final_answer": doc.get("final_answer", "")
        }
        docs.append(payload)
    _docs[:] = docs
    _id_to_payload.clear()
    for i, d in enumerate(_docs):
        _id_to_payload[i] = d
    return _docs


def build_faiss_index():
    global _faiss_index, _docs, _id_to_payload
    if faiss is None or SentenceTransformer is None:
        logger.warning("faiss or sentence-transformers not installed; KB disabled.")
        _faiss_index = None
        return
    _docs = _load_kb()
    if not _docs:
        _faiss_index = None
        return
    model = _get_embed_model()
    texts = [ (d.get("question","") or "") + "\n" + (d.get("final_answer","") or "") for d in _docs ]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    vectors = vectors / norms
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype('float32'))
    _faiss_index = index
    _id_to_payload = {i: _docs[i] for i in range(len(_docs))}
    logger.info("Built FAISS index with %d documents", len(_docs))


def search_kb(query: str, top_k: int = 3) -> List[Dict]:
    global _faiss_index
    if faiss is None or SentenceTransformer is None:
        return []
    if _faiss_index is None:
        try:
            build_faiss_index()
        except Exception as e:
            logger.exception("Failed building FAISS index: %s", e)
            _faiss_index = None
            return []
    if _faiss_index is None:
        return []
    model = _get_embed_model()
    qv = model.encode([query], convert_to_numpy=True)[0]
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    D, I = _faiss_index.search(np.array([qv], dtype='float32'), top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0: continue
        payload = _id_to_payload.get(int(idx), {})
        hits.append({"score": float(score), "payload": payload, "doc_id": int(idx)})
    return hits


# --------------------- HF fallback generator (lazy) ---------------------
def _get_hf_generator():
    global _hf_gen
    if _hf_gen is not None:
        return _hf_gen
    if pipeline is None:
        raise RuntimeError("transformers pipeline not available.")
    try:
        _hf_gen = pipeline("text2text-generation", model=HF_MODEL, device_map=None)
    except Exception:
        _hf_gen = pipeline("text2text-generation", model=HF_MODEL, device_map=None)
    return _hf_gen


def synthesize_with_hf(query: str, sources: List[Dict]) -> Dict:
    gen = None
    try:
        gen = _get_hf_generator()
    except Exception as e:
        logger.exception("HF generator unavailable: %s", e)
        return {"steps": ["MODEL_UNAVAILABLE"], "final_answer": "", "sources": [], "confidence": 0.0}

    src_text = ""
    for s in (sources or [])[:5]:
        src_text += f"- {s.get('title','')}\n  {s.get('url')}\n  {s.get('snippet','')}\n"
    prompt = (
        "You are a math professor. Answer the question step-by-step and give a final concise answer.\n\n"
        f"Question: {query}\n\n"
        f"Sources:\n{src_text}\n\n"
        "Return the answer as:\nSTEPS:\n1. step one\n2. step two\n...\nFINAL: one-line final answer\n"
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
                line = line.lstrip("0123456789. )\t")
                steps.append(line.strip())
        except Exception:
            steps = [out.strip()]
    else:
        steps = [out.strip()]
    return {"steps": steps, "final_answer": final, "sources": [s.get("url") for s in (sources or [])], "confidence": 0.6}


# --------------------- SymPy math solver (robust + plain arithmetic) ---------------------
def _normalize_expr_str(s: str) -> str:
    if not s:
        return s
    s = s.replace("^", "**")
    s = s.replace("²", "**2").replace("³", "**3").replace("⁴", "**4")
    s = s.replace(",", "")
    return s.strip()


def math_solver(query: str) -> Optional[Dict]:
    """
    Parse simple math queries and return deterministic SymPy solution dict.
    Recognizes integrals (definite/indefinite), derivatives, solve, simplify, and plain arithmetic expressions.
    Returns None if query is not recognized as math or SymPy not available.
    """
    if parse_expr is None:
        return None  # SymPy not available
    q = (query or "").strip()
    if not q:
        return None
    ql = q.lower()

    # intent detection (improved)
    is_integral = bool(re.search(r'\bintegral\b|\bintegrate\b|∫', ql))
    is_derivative = bool(re.search(r'\bderivative\b|\bdifferentiate\b|\bd/d', ql))
    is_solve = bool(re.search(r'\bsolve\b|\bfind the roots\b|\bsolve for\b', ql))
    is_simplify = bool(re.search(r'\bsimplify\b|\bsimplification\b', ql))

    # detect plain numeric/math expressions like "39+71" or "(2+3)*5" (allow whitespace and x/X variables)
    is_expr_only = bool(re.match(r'^[\d\s\.\+\-\*\/\^\(\)xX]+$', q))

    # evaluate detection: explicit evaluate keywords OR plain-expression detection
    is_eval = (bool(re.search(r'\bevaluate\b|\bcompute\b|\bwhat is\b', ql) and re.search(r'[0-9x\+\-\*\/\^\(\)]', ql))) or is_expr_only

    try:
        # INTEGRAL
        if is_integral:
            m = re.search(r'integrat(?:e|ion)\s+(?:of\s+)?(.+?)\s+from\s+([-\w\.\*\^]+)\s+to\s+([-\w\.\*\^]+)', q, flags=re.I)
            if not m:
                m = re.search(r'(.+?)\s+d([a-zA-Z])(?:\s+from\s+([-\w\.\*\^]+)\s+to\s+([-\w\.\*\^]+))', q, flags=re.I)
            if m:
                expr_str = m.group(1).strip()
                expr_str = _normalize_expr_str(expr_str)
                var = "x"
                var_m = re.search(r'd([a-zA-Z])', q)
                if var_m:
                    var = var_m.group(1)
                bounds = re.search(r'from\s+([-\w\.\*\^]+)\s+to\s+([-\w\.\*\^]+)', q, flags=re.I)
                x = symbols(var)
                parsed_expr = None
                for parser_try in ("parse_expr", "sympify"):
                    try:
                        if parser_try == "parse_expr":
                            parsed_expr = parse_expr(expr_str, transformations=_transformations)
                        else:
                            parsed_expr = sympify(expr_str)
                        break
                    except Exception:
                        parsed_expr = None
                if parsed_expr is None:
                    return None
                if bounds:
                    a_str = _normalize_expr_str(bounds.group(1))
                    b_str = _normalize_expr_str(bounds.group(2))
                    try:
                        a = parse_expr(a_str, transformations=_transformations)
                    except Exception:
                        a = sympify(a_str)
                    try:
                        b = parse_expr(b_str, transformations=_transformations)
                    except Exception:
                        b = sympify(b_str)
                    antideriv = integrate(parsed_expr, x)
                    val = integrate(parsed_expr, (x, a, b))
                    steps = [
                        f"Parse integrand: {str(parsed_expr)}.",
                        f"Compute antiderivative F({var}) = {str(antideriv)}.",
                        f"Evaluate F({b}) - F({a}) = {str(val)}."
                    ]
                    return {"request_id": "sympy-"+str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(simplify(val)), "sources": ["sympy://definite_integral"], "confidence": 0.99}
                else:
                    antideriv = integrate(parsed_expr, x)
                    steps = [f"Parse integrand: {str(parsed_expr)}.", f"An antiderivative is F({var}) = {str(antideriv)} (indefinite integral = F({var}) + C)."]
                    return {"request_id": "sympy-"+str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(antideriv) + " + C", "sources": ["sympy://indefinite_integral"], "confidence": 0.99}

        # DERIVATIVE
        if is_derivative:
            m = re.search(r'deriv(?:ative)?(?: of)?\s+(.+?)(?:\s+with respect to\s+([a-zA-Z]))?$', q, flags=re.I)
            if not m:
                m = re.search(r'd/d([a-zA-Z])\s+(.+)', q, flags=re.I)
                if m:
                    var = m.group(1)
                    expr_str = m.group(2)
                    m = True
            if m:
                expr_str = m.group(1).strip() if hasattr(m, "group") else expr_str
                expr_str = _normalize_expr_str(expr_str)
                var = (m.group(2) if m and m.groups() and m.group(2) else 'x')
                x = symbols(var)
                parsed_expr = None
                for parser_try in ("parse_expr", "sympify"):
                    try:
                        if parser_try == "parse_expr":
                            parsed_expr = parse_expr(expr_str, transformations=_transformations)
                        else:
                            parsed_expr = sympify(expr_str)
                        break
                    except Exception:
                        parsed_expr = None
                if parsed_expr is None:
                    return None
                deriv = diff(parsed_expr, x)
                steps = [f"Parse function: {str(parsed_expr)}.", f"Compute derivative d/d{var} = {str(deriv)}."]
                return {"request_id": "sympy-"+str(uuid.uuid4())[:8], "steps": steps, "final_answer": str(deriv), "sources": ["sympy://derivative"], "confidence": 0.99}

        # SOLVE
        if is_solve:
            m = re.search(r'solve\s+(.+?)\s*=\s*(.+)', q, flags=re.I)
            if m:
                left_s = _normalize_expr_str(m.group(1).strip())
                right_s = _normalize_expr_str(m.group(2).strip())
                try:
                    left = parse_expr(left_s, transformations=_transformations)
                except Exception:
                    left = sympify(left_s)
                try:
                    right = parse_expr(right_s, transformations=_transformations)
                except Exception:
                    right = sympify(right_s)
                eq = Eq(left, right)
                syms = list(eq.free_symbols)
                var = syms[0] if syms else symbols('x')
                sols = solve(eq, var)
                steps = [f"Equation: {str(eq)}.", f"Solve for {str(var)} -> {str(sols)}."]
                return {"request_id":"sympy-"+str(uuid.uuid4())[:8], "steps":steps, "final_answer":str(sols), "sources":["sympy://solve_equation"], "confidence":0.99}

        # SIMPLIFY / EVALUATE (includes plain arithmetic)
        if is_simplify or is_eval:
            m = re.search(r'(simplify|evaluate|compute|what is)\s+(.*)', q, flags=re.I)
            expr_str = (m.group(2).strip() if m else q)
            expr_str = _normalize_expr_str(expr_str).rstrip(" ?")
            parsed_expr = None
            try:
                parsed_expr = parse_expr(expr_str, transformations=_transformations)
            except Exception:
                try:
                    parsed_expr = sympify(expr_str)
                except Exception:
                    parsed_expr = None
            if parsed_expr is not None:
                simplified = simplify(parsed_expr)
                # convert to plain string (SymPy numbers become Integer/Float; str() is fine)
                steps = [f"Parse expression: {str(parsed_expr)}.", f"Simplify -> {str(simplified)}."]
                return {"request_id":"sympy-"+str(uuid.uuid4())[:8], "steps":steps, "final_answer":str(simplified), "sources":["sympy://simplify"], "confidence":0.99}

    except Exception as e:
        logger.exception("math_solver error: %s", e)
        return None

    return None


# --------------------- Top-level handler ---------------------
def handle_query(query: str, requester: str = "anon") -> Dict:
    """
    Pipeline: 1) SymPy deterministic solver  2) KB retrieval (FAISS)  3) Web+HF fallback
    """
    # 0) Try SymPy math solver first
    try:
        math_res = math_solver(query)
        if math_res is not None:
            return math_res
    except Exception:
        logger.exception("math_solver crashed, falling back to RAG")

    # 1) KB lookup
    try:
        hits = search_kb(query, top_k=3)
        if hits:
            top = hits[0]
            score = float(top.get("score", 0.0))
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                payload = top.get("payload", {})
                return {
                    "request_id": f"kb-{top.get('doc_id')}",
                    "steps": payload.get("steps", []),
                    "final_answer": payload.get("final_answer", ""),
                    "sources": [f"kb://{payload.get('original_id') or top.get('doc_id')}"],
                    "confidence": 0.95
                }
    except Exception:
        logger.exception("KB lookup failed; continuing to web fallback")

    # 2) KB miss -> web search + HF generation
    sources = mcp_search_and_extract(query, top_k=5)
    synthesis = synthesize_with_hf(query, sources)
    synthesis["request_id"] = "mcp-" + (requester or "anon")
    if (not sources) and synthesis.get("confidence", 0.0) == 0.0:
        synthesis.setdefault("steps", []).append("NOTE: No authoritative online sources found; human review recommended.")
    return synthesis


# --------------------- Compatibility RouterAgent shim ---------------------
class RouterAgent:
    """
    Backwards-compatible adapter so code that imports RouterAgent from myagent.rag continues to work.
    Forwards to handle_query.
    """
    def __init__(self, *args, **kwargs):
        pass

    def handle_query(self, query: str, requester: str = None):
        try:
            return handle_query(query, requester=requester)
        except Exception:
            # best-effort fallback
            try:
                m = math_solver(query)
                if m is not None:
                    return m
            except Exception:
                pass
            return {"request_id": "compat-" + str(uuid.uuid4())[:8], "steps": ["No handler available"], "final_answer": "", "sources": [], "confidence": 0.0}

    def ask(self, query: str, requester: str = None):
        return self.handle_query(query, requester=requester)

    def route(self, query: str, requester: str = None):
        return self.handle_query(query, requester=requester)


# --------------------- CLI quick test ---------------------
if __name__ == "__main__":
    print("RAG module quick self-check")
    print("SymPy available:", parse_expr is not None)
    print("duckduckgo available:", _ddg_mod is not None)
    # math test
    try:
        print("math:", math_solver("Integrate x^2 from 0 to 1"))
        print("arith:", math_solver("39+71"))
    except Exception as e:
        print("self-test error:", e)
