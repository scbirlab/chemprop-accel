"""Activation, deactivation, idempotency and fail-safe behaviour of the patch layer."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch
from chemprop.nn.message_passing import mixins

import chemprop_accel as ca
from chemprop_accel import _compat, _patch
from conftest import accel


def _orig_fn():
    with accel(False):
        return mixins._BondMessagePassingMixin.__dict__["message"]


def test_enable_disable_cycles_restore_identity():
    ca.disable()
    original = mixins._BondMessagePassingMixin.__dict__["message"]
    for _ in range(5):
        st = ca.enable()
        assert st.enabled and mixins._BondMessagePassingMixin.__dict__["message"] is not original
        ca.enable()  # idempotent: must not wrap twice
        assert mixins._BondMessagePassingMixin.__dict__["message"].__wrapped__ is original
        st = ca.disable()
        assert not st.enabled and mixins._BondMessagePassingMixin.__dict__["message"] is original


def test_public_class_identity_and_names_unchanged():
    import chemprop.nn as cn

    before = (cn.BondMessagePassing, cn.AtomMessagePassing, cn.MeanAggregation)
    ca.disable()
    ca.enable()
    assert before == (cn.BondMessagePassing, cn.AtomMessagePassing, cn.MeanAggregation)
    assert cn.BondMessagePassing.message.__name__ == "message"


def test_status_reports_everything():
    st = ca.status()
    text = str(st)
    assert st.chemprop_supported and st.enabled
    assert "BondMessagePassing.message" in st.patches and "native extension" in text


def test_no_global_side_effects():
    ca.disable()
    torch.manual_seed(123)
    before = (
        torch.get_num_threads(),
        torch.get_default_dtype(),
        torch.are_deterministic_algorithms_enabled(),
        torch.get_rng_state().clone(),
        torch.get_num_interop_threads(),
    )
    ca.enable()
    after = (
        torch.get_num_threads(),
        torch.get_default_dtype(),
        torch.are_deterministic_algorithms_enabled(),
        torch.get_rng_state().clone(),
        torch.get_num_interop_threads(),
    )
    assert (
        before[0] == after[0]
        and before[1] == after[1]
        and before[2] == after[2]
        and before[4] == after[4]
    )
    assert torch.equal(before[3], after[3]), "enable() must not consume global RNG state"


def test_unknown_fingerprint_is_skipped(monkeypatch):
    ca.disable()
    monkeypatch.setitem(
        _compat.FINGERPRINTS, "BondMessagePassing.message", frozenset({"deadbeef0000"})
    )
    st = ca.enable()
    assert st.patches["BondMessagePassing.message"].startswith("skipped: unrecognised")
    assert st.patches["AtomMessagePassing.message"] == "active"
    ca.disable()


def test_unsupported_chemprop_version_does_nothing(monkeypatch):
    ca.disable()
    monkeypatch.setattr(_compat, "chemprop_version", lambda: "1.6.1")
    st = ca.enable()
    assert not st.enabled and "outside the tested range" in st.reason
    assert mixins._BondMessagePassingMixin.__dict__["message"].__name__ == "message"
    assert not hasattr(mixins._BondMessagePassingMixin.__dict__["message"], "__wrapped__")


def test_failed_selfcheck_rolls_back(monkeypatch):
    ca.disable()
    original = mixins._BondMessagePassingMixin.__dict__["message"]
    _patch.apply(lambda active: "synthetic mismatch")
    assert not _patch.STATE.enabled and "self-check failed" in _patch.STATE.reason
    assert mixins._BondMessagePassingMixin.__dict__["message"] is original
    assert all(p.state != "active" for p in _patch.STATE.patches)


def test_disable_warns_if_wrapped_by_someone_else():
    ca.enable()
    cls = mixins._BondMessagePassingMixin
    ours = cls.__dict__["message"]
    cls.message = lambda self, H, bmg: ours(self, H, bmg)  # a third party re-wraps
    try:
        with pytest.warns(RuntimeWarning, match="re-wrapped"):
            ca.disable()
    finally:
        cls.message = ours
        ca.disable()
        ca.enable()


def _run_py(code, **env):
    e = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=e,
        timeout=300,
    )


def test_env_var_prevents_activation():
    r = _run_py("import chemprop_accel as c; print(c.status().enabled)", CHEMPROP_ACCEL_DISABLE="1")
    assert r.stdout.strip().endswith("False"), r.stderr
    r = _run_py("import chemprop_accel as c; print(c.status().enabled)")
    assert r.stdout.strip().endswith("True"), r.stderr


@pytest.mark.parametrize("order", ["accel_first", "chemprop_first"])
def test_import_order_either_way(order):
    imports = (
        "import chemprop_accel, chemprop"
        if order == "accel_first"
        else "import chemprop, chemprop_accel"
    )
    r = _run_py(f"{imports}\nprint(chemprop_accel.status().enabled)")
    assert r.stdout.strip().endswith("True"), r.stderr


def test_pure_python_fallback_when_native_disabled():
    code = """
    import chemprop_accel as c, torch
    from chemprop import nn
    from chemprop.data import BatchMolGraph
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem
    s = c.status(); print(s.enabled, s.native_available)
    f = SimpleMoleculeMolGraphFeaturizer()
    bmg = BatchMolGraph([f(Chem.MolFromSmiles(x)) for x in ["CCO", "c1ccccc1"]])
    print(nn.BondMessagePassing()(bmg).shape)
    """
    r = _run_py(code, CHEMPROP_ACCEL_NO_NATIVE="1")
    assert "True False" in r.stdout and "torch.Size" in r.stdout, r.stdout + r.stderr


def test_fingerprint_table_covers_installed_version():
    st = ca.status()
    assert all(v == "active" for v in st.patches.values()), st


def test_accelerated_ops_are_actually_called(monkeypatch):
    """Guards against a silent no-op: the fused ops must be hit during a real forward/backward."""
    from chemprop import models, nn

    from conftest import make_batches

    calls = {"edge": 0, "scatter": 0}
    real_e, real_s = ca._ops.edge_message, ca._ops.scatter_sum

    def e(*a, **k):
        calls["edge"] += 1
        return real_e(*a, **k)

    def s(*a, **k):
        calls["scatter"] += 1
        return real_s(*a, **k)

    monkeypatch.setattr(ca._ops, "edge_message", e)
    monkeypatch.setattr(ca._ops, "scatter_sum", s)
    m = models.MPNN(
        nn.BondMessagePassing(d_h=16, depth=4), nn.MeanAggregation(), nn.RegressionFFN(input_dim=16)
    )
    b = make_batches(1)[0]
    m(b.bmg).sum().backward()
    assert (
        calls["edge"] == 3 and calls["scatter"] == 2
    )  # depth-1 messages; readout + mean aggregation
    ca.disable()
    calls.update(edge=0, scatter=0)
    m(b.bmg)
    assert calls == {"edge": 0, "scatter": 0}
    ca.enable()
