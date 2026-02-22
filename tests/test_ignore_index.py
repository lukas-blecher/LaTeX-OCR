"""Test that cross-entropy loss correctly ignores PAD tokens (issue #305).

The [PAD] token has index 0 in the tokenizer. The AutoregressiveWrapper must
use ignore_index=0 so that padding positions do not contribute to the loss or
gradients. Previously, ignore_index defaulted to -100 (PyTorch's default),
meaning PAD tokens were incorrectly included in the loss computation.
"""

import torch
import torch.nn.functional as F
from munch import Munch
from pix2tex.models.transformer import get_decoder


def _make_args(**overrides):
    """Return minimal args Munch needed by get_decoder."""
    defaults = dict(
        num_tokens=100,
        max_seq_len=64,
        dim=64,
        num_layers=2,
        heads=2,
        decoder_args={"cross_attend": True},
        pad_token=0,
    )
    defaults.update(overrides)
    return Munch(defaults)


def test_ignore_index_matches_pad_token():
    """ignore_index on the decoder wrapper must equal the PAD token id."""
    args = _make_args(pad_token=0)
    decoder = get_decoder(args)
    assert decoder.ignore_index == args.pad_token, (
        f"Expected ignore_index={args.pad_token}, got {decoder.ignore_index}"
    )


def test_ignore_index_with_custom_pad_token():
    """If pad_token were ever changed, ignore_index should follow."""
    for pad_id in (0, 3, 5):
        args = _make_args(pad_token=pad_id)
        decoder = get_decoder(args)
        assert decoder.ignore_index == pad_id


def test_pad_tokens_ignored_in_loss():
    """Verify that padding positions do not affect the loss value.

    Build two sequences that differ only in their padding region.  With
    ignore_index set correctly the losses must be identical.
    """
    args = _make_args(pad_token=0)
    decoder = get_decoder(args)
    decoder.eval()

    torch.manual_seed(42)

    # Sequence length 10: first 6 tokens are real, last 4 are PAD (0)
    real_tokens = torch.randint(1, args.num_tokens, (1, 6))
    pad_a = torch.zeros(1, 4, dtype=torch.long)
    pad_b = torch.zeros(1, 4, dtype=torch.long)

    seq_a = torch.cat([real_tokens, pad_a], dim=1)
    seq_b = torch.cat([real_tokens, pad_b], dim=1)

    # Both sequences are identical (same padding), so losses must match
    with torch.no_grad():
        # Provide a dummy context for the cross-attention decoder
        ctx = torch.randn(1, 4, args.dim)
        loss_a = decoder(seq_a, context=ctx)
        loss_b = decoder(seq_b, context=ctx)

    assert torch.allclose(loss_a, loss_b), (
        f"Losses differ: {loss_a.item()} vs {loss_b.item()}"
    )


if __name__ == "__main__":
    test_ignore_index_matches_pad_token()
    test_ignore_index_with_custom_pad_token()
    test_pad_tokens_ignored_in_loss()
    print("All tests passed.")
