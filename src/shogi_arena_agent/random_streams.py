from __future__ import annotations

import hashlib
import random


def child_seed(seed: int | None, *parts: int | str) -> int | None:
    if seed is None:
        return None
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(seed).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("ascii"))
    return int.from_bytes(digest.digest(), "big") & ((1 << 63) - 1)


def child_random(seed: int | None, *parts: int | str) -> random.Random:
    return random.Random(child_seed(seed, *parts))
