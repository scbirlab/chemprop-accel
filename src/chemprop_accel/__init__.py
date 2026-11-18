"""Drop-in multicore CPU acceleration for Chemprop v2.

Usage::

    import chemprop_accel   # activates automatically (set CHEMPROP_ACCEL_DISABLE=1 to prevent)
    import chemprop

See :func:`status`, :func:`enable` and :func:`disable`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import _compat, _native, _ops, _patch, _selfcheck

try:
    from importlib.metadata import version as _version

    __version__ = _version("chemprop-accel")
except Exception:  # noqa: BLE001
    __version__ = "0.0.0+unknown"

__all__ = ["enable", "disable", "status", "Status", "set_variant", "__version__"]

_native.load()
set_variant = _ops.set_variant


@dataclass(frozen=True)
class Status:
    """Snapshot returned by :func:`status`."""

    enabled: bool
    reason: str
    chemprop_version: str | None
    chemprop_supported: bool
    chemprop_supported_range: str
    torch_version: str
    native_available: bool
    native_reason: str
    native_built_against: str | None
    variant: int
    patches: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"chemprop_accel {__version__}: {'ENABLED' if self.enabled else 'disabled'} ({self.reason})",
            f"  chemprop: {self.chemprop_version} (supported {self.chemprop_supported_range}: "
            f"{self.chemprop_supported})",
            f"  torch: {self.torch_version}",
            f"  native extension: {'available' if self.native_available else 'unavailable'}"
            f" - {self.native_reason}"
            + (
                f" [built against torch {self.native_built_against}]"
                if self.native_built_against
                else ""
            ),
            f"  native variant: {self.variant}",
        ]
        lines += [f"  patch {k}: {v}" for k, v in self.patches.items()]
        return "\n".join(lines)


def enable() -> Status:
    """Apply the optimisations (idempotent). Unsupported setups are left untouched and reported."""
    _patch.apply(_selfcheck.run)
    return status()


def disable() -> Status:
    """Restore the original Chemprop methods. Model parameters and state are never touched."""
    for w in _patch.revert():
        import warnings

        warnings.warn(w, RuntimeWarning, stacklevel=2)
    return status()


def status() -> Status:
    import torch

    v = _compat.chemprop_version()
    ok, _ = _compat.version_supported(v)
    lo = ".".join(map(str, _compat.MIN_CHEMPROP))
    hi = ".".join(map(str, _compat.MAX_CHEMPROP_EXCLUSIVE))
    return Status(
        enabled=_patch.STATE.enabled,
        reason=_patch.STATE.reason,
        chemprop_version=v,
        chemprop_supported=ok,
        chemprop_supported_range=f">={lo},<{hi}",
        torch_version=torch.__version__,
        native_available=_native.AVAILABLE,
        native_reason=_native.REASON,
        native_built_against=_native.BUILT_AGAINST,
        variant=_ops.VARIANT,
        patches={p.name: p.state for p in _patch.STATE.patches},
    )


if os.environ.get("CHEMPROP_ACCEL_DISABLE", "") not in ("1", "true", "True", "yes"):
    enable()
