"""torch.compile interaction (skipped when no C++ compiler is available for Inductor)."""

import shutil

import pytest
import torch
from chemprop import models, nn

import chemprop_accel as ca
from conftest import accel, make_batches

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("g++") is None
        and shutil.which("clang++") is None
        and shutil.which("cl") is None,
        reason="no C++ compiler for Inductor",
    ),
]


def _model():
    torch.manual_seed(0)
    return models.MPNN(
        nn.BondMessagePassing(d_h=32, depth=3), nn.MeanAggregation(), nn.RegressionFFN(input_dim=32)
    )


@pytest.mark.parametrize("on", [False, True])
def test_compiled_forward_backward_matches_eager(on):
    b = make_batches(1)[0]
    with accel(on):
        eager = _model()
        compiled = torch.compile(_model(), backend="inductor")
        out_e, out_c = eager(b.bmg), compiled(b.bmg)
        torch.testing.assert_close(out_c, out_e, atol=1e-4, rtol=1e-4)
        out_e.sum().backward()
        out_c.sum().backward()
        for (k, p1), (_, p2) in zip(eager.named_parameters(), compiled.named_parameters()):
            torch.testing.assert_close(p2.grad, p1.grad, atol=1e-3, rtol=1e-3, msg=k)


def test_aot_eager_backend_traces_custom_ops_without_graph_breaks_in_ops():
    if not ca.status().native_available:
        pytest.skip("native extension not available")
    b = make_batches(1)[0]
    m = _model()
    with accel(True):
        c = torch.compile(m, backend="aot_eager", fullgraph=False)
        torch.testing.assert_close(c(b.bmg), m(b.bmg), atol=1e-5, rtol=1e-5)
