
import argparse
import json
import time
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from uuid import uuid4
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import pandas as pd

# sympy for math equivalence checks
try:
    from sympy import sympify, simplify, N, nsimplify
except Exception:
    sympify = None

# ---- Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark")

# ---- Tokenization & metrics
RE_NUMBER = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
SEP_RE = re.compile(r'[\s\(\)\[\]\]\{\},;]+')

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\u2212", "-").replace("^", "**")
    s = s.replace("²", "**2").replace("³", "**3").replace("⁴", "**4")
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[“”"\'`]', '', s)
    return s.strip().lower()

def tokens_of(s: str) -> List[str]:
    s = normalize_text(s)
    toks = [t for t in SEP_RE.split(s) if t]
    return toks

def token_f1(a: str, b: str) -> float:
    ta = tokens_of(a)
    tb = tokens_of(b)
    if not ta or not tb:
        return 0.0
    freq = {}
    for t in ta:
        freq[t] = freq.get(t, 0) + 1
    matched = 0
    for t in tb:
        if freq.get(t, 0) > 0:
            matched += 1
            freq[t] -= 1
    prec = matched / len(tb)
    rec = matched / len(ta)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

# ---- Numeric / math equivalence (robust)
import ast
from math import isclose

def _try_sympy_equal(a: str, b: str, tol: float = 1e-7) -> bool:
    if sympify is None:
        return False
    try:
        A = sympify(a)
        B = sympify(b)
        diff = simplify(A - B)
        # exact symbolic equality
        if diff == 0:
            return True
        # numeric fallback
        na = float(N(A))
        nb = float(N(B))
        return isclose(na, nb, rel_tol=tol, abs_tol=tol)
    except Exception:
        return False

def _parse_sequence(s: str):
    if not s:
        return None
    s0 = s.strip()
    # try python literal
    try:
        val = ast.literal_eval(s0)
        if isinstance(val, (list, tuple, set)):
            return list(val)
    except Exception:
        pass
    # fallback split on commas/spaces when looks like sequence
    if ',' in s0 or ' ' in s0:
        parts = [p.strip() for p in re.split(r'[,\s]+', s0.strip("[]()")) if p.strip()]
        if len(parts) > 1:
            return parts
    return None

def numeric_equivalence(pred: str, gold: str, tol: float = 1e-7) -> bool:
    p = normalize_text(pred)
    g = normalize_text(gold)
    # try sympy scalar/expression equality
    if _try_sympy_equal(p, g, tol=tol):
        return True
    # try sequences (unordered comparison tolerant to numeric equivalence)
    p_seq = _parse_sequence(p)
    g_seq = _parse_sequence(g)
    if p_seq is not None and g_seq is not None:
        # match each gold item to a predicted item (order-insensitive)
        used = [False] * len(p_seq)
        for ge in g_seq:
            matched = False
            for i, pe in enumerate(p_seq):
                if used[i]:
                    continue
                if _try_sympy_equal(str(pe), str(ge), tol=tol):
                    used[i] = True
                    matched = True
                    break
                if normalize_text(str(pe)) == normalize_text(str(ge)):
                    used[i] = True
                    matched = True
                    break
            if not matched:
                return False
        return True
    # fallback: try numeric extraction (first number)
    pm = RE_NUMBER.search(p)
    gm = RE_NUMBER.search(g)
    if pm and gm:
        try:
            pv = float(pm.group(0))
            gv = float(gm.group(0))
            return isclose(pv, gv, rel_tol=tol, abs_tol=tol)
        except Exception:
            pass
    return False

# ---- HTTP helpers: session with retries
def create_session(retries: int = 3, backoff_factor: float = 0.5, status_forcelist=(500,502,503,504)):
    session = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist, allowed_methods=frozenset(['GET','POST']))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# ---- Single-query runner (thread worker)
def run_query(session: requests.Session, host: str, item: dict, timeout: float = 15.0, tries: int = 1) -> Dict[str, Any]:
    qid = item.get("id") or item.get("qid") or str(uuid4())
    q = item.get("question") or item.get("q") or item.get("prompt") or ""
    expected = item.get("answer") or item.get("final_answer") or ""
    url = host.rstrip('/') + '/ask'
    payload = {"query": q}
    t0 = time.perf_counter()
    last_err = ""
    resp_json = None
    for attempt in range(tries):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            elapsed = time.perf_counter() - t0
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                continue
            try:
                resp_json = r.json()
            except Exception:
                # fallback parse text
                resp_json = {"final_answer": r.text}
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(0.1 * (2 ** attempt))
            continue
    elapsed = time.perf_counter() - t0
    if resp_json is None:
        return {
            "id": qid, "question": q, "gold": expected, "pred": "", "confidence": 0.0,
            "time_sec": elapsed, "exact_match": False, "numeric_match": False, "token_f1": 0.0, "error": last_err
        }
    # normalize response extraction
    pred = ""
    confidence = 0.0
    if isinstance(resp_json, dict):
        pred = resp_json.get("final_answer")
        if not pred:
            steps = resp_json.get("steps")
            if isinstance(steps, list) and steps:
                pred = " ".join(str(x) for x in steps)
            else:
                pred = resp_json.get("answer") or resp_json.get("text") or json.dumps(resp_json, ensure_ascii=False)
        try:
            confidence = float(resp_json.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
    else:
        pred = str(resp_json)
    pred = str(pred).strip()
    gold = str(expected).strip()
    em = normalize_text(pred) == normalize_text(gold)
    nm = numeric_equivalence(pred, gold)
    tf1 = token_f1(pred, gold)
    return {
        "id": qid, "question": q, "gold": gold, "pred": pred, "confidence": confidence,
        "time_sec": elapsed, "exact_match": em, "numeric_match": nm, "token_f1": tf1, "error": ""
    }

# ---- Batch runner
def run_benchmark(host: str, items: List[Dict], out_csv: str, out_jsonl: str = None,
                  workers: int = 8, timeout: float = 15.0, retries: int = 3, backoff: float = 0.5):
    session = create_session(retries=retries, backoff_factor=backoff)
    results = []
    run_one = partial(run_query, session, host, timeout=timeout, tries=retries)
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(run_one, item): item for item in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Running benchmark", unit="q"):
            try:
                res = fut.result()
            except Exception as e:
                logger.exception("Worker exception: %s", e)
                res = {"id": None, "question": str(futures.get(fut)), "gold": "", "pred": "", "confidence": 0.0,
                       "time_sec": 0.0, "exact_match": False, "numeric_match": False, "token_f1": 0.0, "error": str(e)}
            results.append(res)
            # write JSONL incrementally if requested (good for large runs)
            if out_jsonl:
                with open(out_jsonl, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(res, ensure_ascii=False) + "\n")
    # dataframe and save
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    # summary
    total = len(df)
    exact = int(df["exact_match"].sum()) if "exact_match" in df.columns else 0
    numeric = int(df["numeric_match"].sum()) if "numeric_match" in df.columns else 0
    mean_f1 = float(df["token_f1"].mean()) if "token_f1" in df.columns else 0.0
    mean_conf = float(df["confidence"].mean()) if "confidence" in df.columns else 0.0
    mean_time = float(df["time_sec"].mean()) if "time_sec" in df.columns else 0.0
    print("\n=== Benchmark summary ===")
    print(f"Total: {total}")
    print(f"Exact match: {exact} ({100.0 * exact / max(1,total):.2f}%)")
    print(f"Numeric match: {numeric} ({100.0 * numeric / max(1,total):.2f}%)")
    print(f"Mean token-F1: {mean_f1:.3f}")
    print(f"Mean confidence: {mean_conf:.3f}")
    print(f"Mean response time: {mean_time:.3f} sec")
    print(f"Results saved: {out_csv}")
    if out_jsonl:
        print(f"Incremental JSONL saved: {out_jsonl}")

# ---- CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, help="Math agent host e.g. http://127.0.0.1:8000")
    p.add_argument("--input", required=True, help="JSON file (list of {id, question, answer})")
    p.add_argument("--out", required=True, help="CSV output path")
    p.add_argument("--out-jsonl", default=None, help="Optional incremental JSONL output")
    p.add_argument("--workers", type=int, default=8, help="Concurrent workers")
    p.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout (sec)")
    p.add_argument("--retries", type=int, default=3, help="Per-request retries")
    p.add_argument("--backoff", type=float, default=0.5, help="Retry backoff factor")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)
    run_benchmark(
        host=args.host, items=items, out_csv=args.out, out_jsonl=args.out_jsonl,
        workers=args.workers, timeout=args.timeout, retries=args.retries, backoff=args.backoff
    )

if __name__ == "__main__":
    main()
