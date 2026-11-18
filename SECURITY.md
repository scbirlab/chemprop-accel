# Security policy

`chemprop-accel` executes native code in-process and monkey-patches a small number of Chemprop methods.

- Report suspected vulnerabilities privately through GitHub's *Security → Report a vulnerability* form
  (private vulnerability reporting), not in public issues.
- Native kernels validate index ranges and tensor dtypes/devices and raise `RuntimeError` rather than reading out of
  bounds. A report showing otherwise is treated as a security bug.
- The benchmark tool performs no network access and writes only to its `--output` directory.
- Only the latest minor release receives fixes.
