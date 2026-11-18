# Installation

```bash
pip install chemprop-accel          # pulls torch; install chemprop yourself (or pip install "chemprop-accel[chemprop]")
```

`chemprop-accel` has two layers:

| Layer | What it needs | Speed |
| --- | --- | --- |
| **Native extension** (`chemprop_accel/_C.*`, C++) | a wheel built for your **PyTorch minor version** and platform, or a C++17 compiler to build one | fastest |
| **Pure-PyTorch fallback** | nothing extra | most of the speedup comes from removing Chemprop's expanded index tensors, see {doc}`optimisations` |

Importing the package never compiles anything. If the native extension is missing or was built against a different
PyTorch minor version, `chemprop_accel` says so in `status()` and uses the fallback.

## Native wheels

Native wheels depend on the PyTorch C++ ABI (one wheel set per torch *minor* version) but **not** on the Python
version, because the library does not link against Python. They are therefore tagged `py3-none-<platform>`. Each
release attaches wheels for the torch versions listed in `.github/workflows/wheels.yml` to its GitHub release; the torch
minor version is encoded in the wheel's *build tag* (`chemprop_accel-X.Y.Z-1torch2.15-py3-none-<platform>.whl`), and pip
cannot choose between them automatically, so download/install the one matching your torch; PyPI
carries the sdist and a pure-Python wheel. Do not assume a wheel exists for your exact torch/platform combination:
check `chemprop_accel.status()`.

## Build from source

```bash
pip install torch                      # first: the extension is compiled against this exact torch
pip install setuptools wheel ninja
CHEMPROP_ACCEL_REQUIRE_NATIVE=1 pip install --no-build-isolation chemprop-accel --no-binary chemprop-accel
```

`--no-build-isolation` is required so the compiler sees your installed PyTorch headers. On x86-64 Linux with GCC the
hot loops are compiled for AVX-512, AVX2 and baseline x86-64 and selected at run time, so a wheel built on one machine
runs on older CPUs. No AVX-512/AMX is required. macOS (arm64/x86-64) and Windows (MSVC) builds use the compiler's
default vectorisation.

Environment variables for the build: `CHEMPROP_ACCEL_NO_NATIVE=1` (pure wheel), `CHEMPROP_ACCEL_REQUIRE_NATIVE=1`
(fail instead of falling back).

## Runtime environment variables

| Variable | Effect |
| --- | --- |
| `CHEMPROP_ACCEL_DISABLE=1` | importing the package does **not** activate the patches |
| `CHEMPROP_ACCEL_NO_NATIVE=1` | never load the native library (use the pure-PyTorch fallback) |
| `CHEMPROP_ACCEL_VARIANT=0|1|2` | native algorithm variant (default 2); for benchmarking |
