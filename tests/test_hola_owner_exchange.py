"""The two invariants that isolate the new information path from HoLa."""
import torch

from src.models.vae.hola_owner import OwnerBoundaryExchange


def test_identity_then_messages_only_from_same_owner():
    torch.manual_seed(42)
    exchange = OwnerBoundaryExchange(4, 8, 3, 2, 1)
    pairs = torch.cartesian_prod(torch.arange(2), torch.arange(2))
    h = torch.randn(4, 4, requires_grad=True)
    geometry = torch.randn(4, 3, requires_grad=True)
    logits = torch.randn(4, requires_grad=True)

    def forward():
        return exchange(h, geometry, logits, pairs, pairs[:, 1], 2, 2)

    torch.testing.assert_close(forward(), h, atol=0, rtol=0)
    with torch.no_grad():
        exchange.readout.weight[:, 4:].fill_(1.0)
    forward()[0].sum().backward()
    assert geometry.grad[1].abs().sum() > 0
    assert geometry.grad[2:].count_nonzero() == 0
    assert logits.grad is None
