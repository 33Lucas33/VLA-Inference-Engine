from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    if not values or not 0 <= q <= 100:
        raise ValueError("values must be nonempty and q in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson interval, adapted from the parent SmolVLA-LIBERO result summarizer."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("invalid successes/total")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return 100 * (center - spread), 100 * (center + spread)
