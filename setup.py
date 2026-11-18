"""Build script for the optional native extension.

The native library has no Python-ABI dependency (it is loaded with ``torch.ops.load_library``),
so wheels are tagged ``py3-none-<platform>``. It *is* tied to the PyTorch minor version it was
compiled against; ``chemprop_accel`` checks this at import time and falls back to pure PyTorch.

Environment variables
---------------------
CHEMPROP_ACCEL_NO_NATIVE=1   skip the native build and produce a pure-Python wheel.
CHEMPROP_ACCEL_REQUIRE_NATIVE=1   fail the build if the extension cannot be compiled.
"""

from __future__ import annotations

import os
import sys

from setuptools import setup

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover
    bdist_wheel = None


def _want_native() -> bool:
    if os.environ.get("CHEMPROP_ACCEL_NO_NATIVE") == "1":
        return False
    try:
        import torch  # noqa: F401

    except ImportError:
        print(
            "chemprop-accel: torch not importable at build time; building pure-Python wheel "
            "(use --no-build-isolation to build the native extension).",
            file=sys.stderr,
        )
        return False
    return True


ext_modules = []
cmdclass = {}
if _want_native():
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    with open(os.path.join("src", "chemprop_accel", "_build_info.py"), "w") as fh:
        fh.write(f'TORCH_BUILT_AGAINST = "{torch.__version__}"\n')

    extra = ["/O2"] if sys.platform == "win32" else ["-O3", "-fvisibility=hidden"]
    ext_modules = [
        CppExtension(
            name="chemprop_accel._C",
            sources=["csrc/graph_ops.cpp"],
            extra_compile_args=extra,
        )
    ]

    class SafeBuildExt(BuildExtension):
        def run(self):
            try:
                super().run()
            except Exception as exc:  # noqa: BLE001
                if os.environ.get("CHEMPROP_ACCEL_REQUIRE_NATIVE") == "1":
                    raise
                print(
                    f"chemprop-accel: native build failed ({exc}); continuing pure-Python.",
                    file=sys.stderr,
                )
                self.extensions = []

        def build_extension(self, ext):
            try:
                super().build_extension(ext)
            except Exception as exc:  # noqa: BLE001
                if os.environ.get("CHEMPROP_ACCEL_REQUIRE_NATIVE") == "1":
                    raise
                print(
                    f"chemprop-accel: native build failed ({exc}); continuing pure-Python.",
                    file=sys.stderr,
                )

    cmdclass["build_ext"] = SafeBuildExt.with_options(no_python_abi_suffix=True)

if bdist_wheel is not None:

    class PlatformWheel(bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = not ext_modules

        def get_tag(self):
            if not ext_modules:
                return "py3", "none", "any"
            _, _, plat = super().get_tag()
            return "py3", "none", plat

    cmdclass["bdist_wheel"] = PlatformWheel

setup(ext_modules=ext_modules, cmdclass=cmdclass)
