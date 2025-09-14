from __future__ import annotations
from typing import Iterable, List, Dict, Any, Callable

# Default K for k-anonymity, can be overridden per-call
K_ANON_DEFAULT = 5


def apply_k_anon(items: Iterable[Dict[str, Any]], count_key: str, k_min: int = K_ANON_DEFAULT) -> List[Dict[str, Any]]:
    """
    Filter out items whose count metric is below k_min.

    Args:
        items: iterable of dict-like objects
        count_key: key in each item that represents the count for k-anon
        k_min: minimum allowed count (inclusive)

    Returns:
        Filtered list with only items having item[count_key] >= k_min
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        try:
            if int(it.get(count_key, 0)) >= int(k_min):
                out.append(it)
        except Exception:
            # If item doesn't have a valid count, suppress it
            continue
    return out
