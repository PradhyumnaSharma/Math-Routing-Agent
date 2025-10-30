
import importlib
import traceback

_ddg_mod = None
try:
    _ddg_mod = importlib.import_module("duckduckgo_search")
except Exception:
    _ddg_mod = None

def _normalize_entry(r):
    # normalize dict keys from different ddg versions
    return {
        "title": r.get("title") or r.get("source") or "",
        "url": r.get("href") or r.get("url") or r.get("link") or "",
        "snippet": r.get("body") or r.get("snippet") or r.get("text") or "",
        "trust_score": 0.6
    }

def mcp_search_and_extract(query: str, top_k: int = 5):
    results = []
    if not _ddg_mod:
        # duckduckgo_search not installed
        return results

    try:
        # Preferred API: ddg(query, max_results=...)
        if hasattr(_ddg_mod, "ddg"):
            raw = _ddg_mod.ddg(query, max_results=top_k) or []
            for r in raw:
                results.append(_normalize_entry(r))
            return results

        # Fallback: search(query, max_results=...)
        if hasattr(_ddg_mod, "search"):
            raw = _ddg_mod.search(query, max_results=top_k) or []
            for r in raw:
                results.append(_normalize_entry(r))
            return results

        # Fallback: ddg_answers(query) -> dict structure
        if hasattr(_ddg_mod, "ddg_answers"):
            try:
                ans = _ddg_mod.ddg_answers(query)
                candidates = []
                if isinstance(ans, dict):
                    # try some known keys
                    for key in ("answers", "related", "results"):
                        if key in ans and isinstance(ans[key], list):
                            candidates = ans[key][:top_k]
                            break
                if isinstance(candidates, list):
                    for a in candidates:
                        if isinstance(a, dict):
                            results.append(_normalize_entry(a))
                return results
            except Exception:
                # ignore and continue
                pass

    except Exception:
        # if anything goes wrong return empty list (caller will handle human-in-loop note)
        # but log to stderr for debugging
        try:
            import sys
            sys.stderr.write("mcp_search_and_extract error:\\n" + traceback.format_exc())
        except Exception:
            pass
        return []

    return results
