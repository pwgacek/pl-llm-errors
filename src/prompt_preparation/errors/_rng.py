from __future__ import annotations

import hashlib


def deterministic_seed(base_seed: int, text: str, salt: str) -> int:
    """Return a stable 64-bit seed derived from the input text and salt."""
    payload = f"{salt}::{text}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    derived = int.from_bytes(digest[:8], "big", signed=False)
    return base_seed ^ derived
