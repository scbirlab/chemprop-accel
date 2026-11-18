"""Version gating and source fingerprints for patch targets.

Chemprop internals are private. Rather than trusting a version range alone, every patch is applied
only if (a) the installed Chemprop version is inside the supported range *and* (b) the normalised
source of the exact function being replaced matches a fingerprint that the replacement has been
verified against. Anything else is left untouched and reported by :func:`chemprop_accel.status`.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from importlib import metadata

#: Inclusive lower / exclusive upper bound of Chemprop versions this release is tested with.
MIN_CHEMPROP = (2, 2, 0)
MAX_CHEMPROP_EXCLUSIVE = (2, 4, 0)

#: Known-good source fingerprints per patch target. ``dev`` entries are the pinned development
#: commit listed in ``CHEMPROP_COMMIT``.
FINGERPRINTS: dict[str, frozenset[str]] = {
    "BondMessagePassing.message": frozenset({"f8259411357b", "fa147f278c03"}),
    "AtomMessagePassing.message": frozenset({"ae3e3d58a45a", "d0561cc0ddfe"}),
    "MessagePassing.forward": frozenset({"4f4b5da5ec4b", "a257702de9ca"}),
    "MeanAggregation.forward": frozenset({"3678c307ad45", "d18c87483574"}),
    "SumAggregation.forward": frozenset({"82c26d47b208"}),
}


def fingerprint(fn) -> str | None:
    """Whitespace-insensitive SHA-256 prefix of ``fn``'s source, or ``None`` if unavailable."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(re.sub(r"\s+", "", src).encode()).hexdigest()[:12]


def parse_version(v: str) -> tuple[int, ...]:
    parts = []
    for tok in re.split(r"[.+\-]", v):
        m = re.match(r"\d+", tok)
        if not m:
            break
        parts.append(int(m.group()))
    return tuple(parts)


def chemprop_version() -> str | None:
    try:
        return metadata.version("chemprop")
    except metadata.PackageNotFoundError:
        return None


def version_supported(v: str | None) -> tuple[bool, str]:
    if v is None:
        return False, "chemprop is not installed"
    t = parse_version(v)
    if not t:
        return False, f"unparseable chemprop version {v!r}"
    if t < MIN_CHEMPROP or t >= MAX_CHEMPROP_EXCLUSIVE:
        lo = ".".join(map(str, MIN_CHEMPROP))
        hi = ".".join(map(str, MAX_CHEMPROP_EXCLUSIVE))
        return False, f"chemprop {v} is outside the tested range [{lo}, {hi})"
    return True, "ok"
